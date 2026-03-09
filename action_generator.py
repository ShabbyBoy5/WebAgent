"""
Candidate action generation using Claude via the Anthropic API.
"""

import os
import re
import json
import base64
import anthropic


def _get_api_key() -> str:
    """Get Anthropic API key from env or Claude Code credentials."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    creds_path = os.path.expanduser("~/.claude/.credentials.json")
    if os.path.exists(creds_path):
        with open(creds_path) as f:
            creds = json.load(f)
        oauth = creds.get("claudeAiOauth", {})
        token = oauth.get("accessToken")
        if token:
            return token
    raise RuntimeError("No ANTHROPIC_API_KEY set and no Claude Code credentials found")


SYSTEM_PROMPT_SINGLE = """You are a web navigation agent. Given a screenshot of a webpage, its interactive elements, and a task, decide the single best next action.

Reply with EXACTLY one line in one of these formats:
type [<id>] "<text>"
click [<id>]
press "<key>"
scroll down
scroll up
go_back
stop

Rules:
- Use type [id] "text" to enter text in a searchbox or textbox, then the system will auto-press Enter
- Use click [id] to click buttons or links
- Reference [id] numbers from the interactive elements list
- Do NOT repeat previous actions
- Only output the action line, nothing else"""

SYSTEM_PROMPT_MULTI = """You are a web navigation agent. Given a screenshot of a webpage, its interactive elements, and a task, propose 3 to 5 candidate next actions ranked from most to least promising.

Use EXACTLY these formats, one per line, numbered:
1. type [<id>] "<text>"
2. click [<id>]
3. scroll down

Available action formats:
type [<id>] "<text>"
click [<id>]
press "<key>"
scroll down
scroll up
go_back
stop

Rules:
- Use type [id] "text" to enter text in a searchbox or textbox, then the system will auto-press Enter
- Use click [id] to click buttons or links
- Reference [id] numbers from the interactive elements list
- Do NOT repeat previous actions
- Output ONLY the numbered action lines, nothing else"""


client = None


def _get_client():
    global client
    if client is None:
        client = anthropic.Anthropic(api_key=_get_api_key())
    return client


def _build_prompt_and_image(screenshot_b64: str, task: str,
                            action_history: list[str],
                            accessibility_tree: str, question: str):
    """Build the user prompt and extract image data."""
    history_str = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(action_history)
    ) if action_history else "  (none)"

    tree_display = accessibility_tree
    if len(tree_display) > 4000:
        tree_display = tree_display[:4000] + "\n... (truncated)"

    user_prompt = f"""Task: {task}

Previous actions:
{history_str}

Interactive elements on this page:
{tree_display}

{question}"""

    if screenshot_b64.startswith("data:image"):
        media_type = screenshot_b64.split(";")[0].split(":")[1]
        b64_data = screenshot_b64.split(",", 1)[1]
    else:
        media_type = "image/png"
        b64_data = screenshot_b64

    return user_prompt, media_type, b64_data


def generate_candidates(screenshot_b64: str, task: str,
                        action_history: list[str],
                        accessibility_tree: str,
                        multi: bool = False) -> list[dict]:
    """Generate candidate actions using Claude.

    If multi=True, asks for 3-5 ranked candidates (for planning mode).
    Otherwise, asks for a single best action (reactive mode).
    """
    if multi:
        system = SYSTEM_PROMPT_MULTI
        question = "Propose 3 to 5 candidate actions, numbered:"
        max_tokens = 512
    else:
        system = SYSTEM_PROMPT_SINGLE
        question = "What is the single best next action?"
        max_tokens = 256

    user_prompt, media_type, b64_data = _build_prompt_and_image(
        screenshot_b64, task, action_history, accessibility_tree, question
    )

    resp = _get_client().messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64_data},
                },
                {"type": "text", "text": user_prompt},
            ],
        }],
    )

    raw_output = resp.content[0].text.strip()
    print(f"  Claude says: {raw_output}")

    if multi:
        return _parse_multiple_actions(raw_output)
    else:
        parsed = _parse_action(raw_output)
        return [parsed] if parsed else []


def _parse_multiple_actions(text: str) -> list[dict]:
    """Parse multiple numbered action lines from Claude's response."""
    results = []
    for line in text.split("\n"):
        line = line.strip().strip("`")
        if not line:
            continue
        # Strip leading number + dot/paren (e.g. "1. ", "2) ")
        cleaned = re.sub(r'^\d+[.)]\s*', '', line)
        if not cleaned:
            continue
        parsed = _parse_single_line(cleaned)
        if parsed:
            # Overwrite raw with the cleaned version
            parsed["raw"] = cleaned
            results.append(parsed)
    return results


def _parse_action(text: str) -> dict | None:
    """Parse a structured action from text, scanning all lines."""
    text = text.strip()
    if not text:
        return None

    # Try each line (Claude sometimes adds explanation before the action)
    for line in text.split("\n"):
        line = line.strip().strip("`")
        if not line:
            continue
        result = _parse_single_line(line)
        if result:
            return result

    return None


def _parse_single_line(line: str) -> dict | None:
    """Parse a single action line."""

    # stop
    if line.lower().startswith("stop"):
        return {"action_type": "stop", "element_id": None, "value": None, "raw": line}

    # scroll down / scroll up
    if re.match(r'scroll\s+(down|up)', line, re.IGNORECASE):
        direction = "scroll down" if "down" in line.lower() else "scroll up"
        return {"action_type": direction, "element_id": None, "value": None, "raw": line}

    # go_back
    if re.match(r'go_?back', line, re.IGNORECASE):
        return {"action_type": "go_back", "element_id": None, "value": None, "raw": line}

    # press "Key"
    m = re.match(r'press\s+"([^"]+)"', line, re.IGNORECASE)
    if m:
        return {"action_type": "press", "element_id": None, "value": m.group(1), "raw": line}

    # type [id] "text"
    m = re.match(r'type\s+\[(\d+)\]\s+"([^"]*)"', line, re.IGNORECASE)
    if m:
        return {"action_type": "type", "element_id": int(m.group(1)), "value": m.group(2), "raw": line}

    # click [id]
    m = re.match(r'click\s+\[(\d+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": "click", "element_id": int(m.group(1)), "value": None, "raw": line}

    # hover [id]
    m = re.match(r'hover\s+\[(\d+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": "hover", "element_id": int(m.group(1)), "value": None, "raw": line}

    # Fallback: any action with [id]
    m = re.match(r'(\w+)\s+\[(\d+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": m.group(1).lower(), "element_id": int(m.group(2)), "value": None, "raw": line}

    return None
