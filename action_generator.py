"""
Candidate action generation using Claude via claud CLI or Anthropic API.
"""

import os
import re
import json
import base64
import subprocess


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
- type [id] "text" types AND auto-presses Enter — this submits searches automatically. Do NOT click a search button after typing.
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
- type [id] "text" types AND auto-presses Enter — this submits searches automatically. Do NOT click a search button after typing.
- Use click [id] to click buttons or links
- Reference [id] numbers from the interactive elements list
- Do NOT repeat previous actions
- Output ONLY the numbered action lines, nothing else"""

def _plan_system_prompt(min_steps: int = 0) -> str:
    min_steps_rule = ""
    if min_steps > 0:
        min_steps_rule = f"\n- The plan MUST have at least {min_steps} action steps (not counting \"stop\")"

    return f"""You are an expert web navigation planner. You will receive a task, the current page's interactive elements, and you must produce a COMPLETE action plan from the current state all the way to task completion.

Think through the task in phases:

PHASE 1 — CURRENT PAGE: What actions can you take right now using the visible [id] elements?
PHASE 2 — NAVIGATION: After Phase 1, what pages will load? What will you need to click/type on those pages?
PHASE 3 — TASK COMPLETION: What final actions confirm the task is done?

Output format — a numbered list where EVERY line is exactly one of:
  type [<id>] "<text>"
  click [<id>]
  press "<key>"
  scroll down
  scroll up
  go_back
  stop

CRITICAL — how "type" works:
- type [id] "text" types the text AND auto-presses Enter. This means typing in a search box automatically submits the search. You do NOT need to click a search button after typing — it is done for you.
- So to search: just use type [id] "query" — that's it. Do NOT add a separate click on a search button.

Important rules:
- For Phase 1: use the exact [id] numbers from the interactive elements list below
- For Phase 2+: you cannot see those pages yet, so use placeholder IDs with words that will likely appear in the element's actual text on the page. Good: [?search_box], [?sort_by_dropdown], [?first_product], [?amazon_link]. Bad: [?highest_rated_cookbook] (too abstract — the element text will be a book title, not "highest rated cookbook"). Use ordinals like [?first_result] when you want the Nth matching element
- Every plan MUST end with "stop" as the final action{min_steps_rule}
- Do NOT skip steps — include every click, every type, every scroll needed
- Do NOT explain or comment — output ONLY the numbered action lines

Example for "find sony headphones on amazon" starting from a search engine:
1. type [1] "amazon.com sony headphones"
2. click [?first_result]
3. type [?search_box] "sony headphones"
4. click [?first_product]
5. scroll down
6. stop"""


def _has_claud_proxy() -> bool:
    """Check if claud CLI is available."""
    import shutil
    return shutil.which("claud") is not None


def _get_claud_env() -> dict:
    """Build environment for claud subprocess with proxy and token."""
    env = {**os.environ}
    proxy_url = (os.environ.get("_SAVED_HTTPS_PROXY")
                 or os.environ.get("HTTPS_PROXY")
                 or os.environ.get("https_proxy")
                 or "https://46.225.188.115:8080")
    env["HTTPS_PROXY"] = proxy_url
    env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
    custom_path = os.path.expanduser("~/.claude-custom")
    if os.path.exists(custom_path):
        with open(custom_path) as f:
            token = f.read().strip()
        if token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
    return env


# Track active subprocess so it can be killed on cancel
_active_proc: subprocess.Popen | None = None
_proc_lock = __import__("threading").Lock()


def kill_active_llm_call():
    """Kill any in-flight claud subprocess. Called by dashboard on cancel."""
    with _proc_lock:
        if _active_proc is not None and _active_proc.poll() is None:
            _active_proc.kill()


def _call_claud(system: str, user_prompt: str) -> str:
    """Call claud CLI with a text-only prompt (no screenshot for speed)."""
    global _active_proc
    full_prompt = f"{system}\n\n{user_prompt}"
    env = _get_claud_env()
    with _proc_lock:
        _active_proc = subprocess.Popen(
            ["claud", "--print", "--model", "claude-haiku-4-5-20251001"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
    try:
        stdout, _ = _active_proc.communicate(input=full_prompt, timeout=60)
    except subprocess.TimeoutExpired:
        _active_proc.kill()
        _active_proc.communicate()
        stdout = ""
    finally:
        with _proc_lock:
            _active_proc = None
    return stdout.strip()


def _call_api(system: str, user_prompt: str, media_type: str, b64_data: str) -> str:
    """Call Anthropic API directly."""
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        creds_path = os.path.expanduser("~/.claude/.credentials.json")
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                creds = json.load(f)
            oauth = creds.get("claudeAiOauth", {})
            key = oauth.get("accessToken")

    if not key:
        raise RuntimeError("No ANTHROPIC_API_KEY or Claude Code credentials found")

    if key.startswith("sk-ant-oat"):
        client = anthropic.Anthropic(
            auth_token=key,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    else:
        client = anthropic.Anthropic(api_key=key)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
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
    return resp.content[0].text.strip()


def _build_prompt(task: str, action_history: list[str],
                  accessibility_tree: str, question: str,
                  session_context: str = "") -> str:
    """Build the user prompt text."""
    history_str = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(action_history)
    ) if action_history else "  (none)"

    tree_display = accessibility_tree
    if len(tree_display) > 4000:
        tree_display = tree_display[:4000] + "\n... (truncated)"

    session_block = f"\n{session_context}\n" if session_context else ""

    return f"""Task: {task}

Previous actions:
{history_str}

Interactive elements on this page:
{tree_display}
{session_block}
{question}"""


def generate_candidates(screenshot_b64: str, task: str,
                        action_history: list[str],
                        accessibility_tree: str,
                        multi: bool = False,
                        session_context: str = "") -> list[dict]:
    """Generate candidate actions using Claude.

    If multi=True, asks for 3-5 ranked candidates (for planning mode).
    Otherwise, asks for a single best action (reactive mode).
    """
    if multi:
        system = SYSTEM_PROMPT_MULTI
        question = "Propose 3 to 5 candidate actions, numbered:"
    else:
        system = SYSTEM_PROMPT_SINGLE
        question = "What is the single best next action?"

    user_prompt = _build_prompt(task, action_history, accessibility_tree, question, session_context)

    if _has_claud_proxy():
        raw_output = _call_claud(system, user_prompt)
    else:
        # Direct API call with screenshot
        if screenshot_b64.startswith("data:image"):
            media_type = screenshot_b64.split(";")[0].split(":")[1]
            b64_data = screenshot_b64.split(",", 1)[1]
        else:
            media_type = "image/png"
            b64_data = screenshot_b64
        raw_output = _call_api(system, user_prompt, media_type, b64_data)

    print(f"  Claude says: {raw_output}")

    if multi:
        return _parse_multiple_actions(raw_output)
    else:
        parsed = _parse_action(raw_output)
        return [parsed] if parsed else []


def generate_full_plans(screenshot_b64: str, task: str,
                        accessibility_tree: str,
                        num_plans: int = 3,
                        min_steps: int = 0,
                        session_context: str = "",
                        event_callback=None) -> list[list[dict]]:
    """Generate multiple complete action plans for a task.

    Returns a list of plans, where each plan is a list of action dicts.
    If event_callback is provided, emits per-step events for streaming.
    """
    plans = []
    system = _plan_system_prompt(min_steps)
    question = f"Produce a complete step-by-step plan to accomplish this task. Plan {len(plans)+1}:"

    for i in range(num_plans):
        if event_callback:
            event_callback("status", {"message": f"Generating plan {i+1}/{num_plans}..."})

        user_prompt = _build_prompt(task, [], accessibility_tree, question, session_context)
        if i > 0:
            prev_summary = ""
            for j, p in enumerate(plans):
                steps_str = "\n".join(f"  {s+1}. {a['raw']}" for s, a in enumerate(p))
                prev_summary += f"\nPlan {j+1} (already generated, do NOT repeat):\n{steps_str}\n"
            user_prompt += f"\n\nPrevious plans (generate a DIFFERENT approach):{prev_summary}"

        if _has_claud_proxy():
            raw_output = _call_claud(system, user_prompt)
        else:
            if screenshot_b64.startswith("data:image"):
                media_type = screenshot_b64.split(";")[0].split(":")[1]
                b64_data = screenshot_b64.split(",", 1)[1]
            else:
                media_type = "image/png"
                b64_data = screenshot_b64
            raw_output = _call_api(system, user_prompt, media_type, b64_data)

        print(f"  Plan {i+1} raw: {raw_output[:200]}...")
        parsed = _parse_multiple_actions(raw_output)
        if parsed:
            plans.append(parsed)
            if event_callback:
                for j, a in enumerate(parsed):
                    event_callback("plan_step", {
                        "plan_index": len(plans) - 1,
                        "step_index": j,
                        "raw": a["raw"],
                        "action_type": a["action_type"],
                    })
                event_callback("plan_complete", {
                    "plan_index": len(plans) - 1,
                    "total_steps": len(parsed),
                })

    print(f"  Generated {len(plans)} complete plans")
    if event_callback:
        event_callback("all_plans_complete", {"count": len(plans)})
    return plans


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
            parsed["raw"] = cleaned
            results.append(parsed)
    return results


def _parse_action(text: str) -> dict | None:
    """Parse a structured action from text, scanning all lines."""
    text = text.strip()
    if not text:
        return None

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

    # type [id] "text" — numeric ID
    m = re.match(r'type\s+\[(\d+)\]\s+"([^"]*)"', line, re.IGNORECASE)
    if m:
        return {"action_type": "type", "element_id": int(m.group(1)), "value": m.group(2), "raw": line}

    # type [?placeholder] "text" — future step placeholder
    m = re.match(r'type\s+\[(\?\w+)\]\s+"([^"]*)"', line, re.IGNORECASE)
    if m:
        return {"action_type": "type", "element_id": m.group(1), "value": m.group(2), "raw": line}

    # click [id] — numeric ID
    m = re.match(r'click\s+\[(\d+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": "click", "element_id": int(m.group(1)), "value": None, "raw": line}

    # click [?placeholder] — future step placeholder
    m = re.match(r'click\s+\[(\?\w+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": "click", "element_id": m.group(1), "value": None, "raw": line}

    # hover [id] — numeric ID
    m = re.match(r'hover\s+\[(\d+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": "hover", "element_id": int(m.group(1)), "value": None, "raw": line}

    # hover [?placeholder]
    m = re.match(r'hover\s+\[(\?\w+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": "hover", "element_id": m.group(1), "value": None, "raw": line}

    # Fallback: any action with [id] (numeric)
    m = re.match(r'(\w+)\s+\[(\d+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": m.group(1).lower(), "element_id": int(m.group(2)), "value": None, "raw": line}

    # Fallback: any action with [?placeholder]
    m = re.match(r'(\w+)\s+\[(\?\w+)\]', line, re.IGNORECASE)
    if m:
        return {"action_type": m.group(1).lower(), "element_id": m.group(2), "value": None, "raw": line}

    return None
