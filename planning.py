"""
Planning pipeline: generate candidates -> predict outcomes with Dreamer-7B -> score with Claude.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from action_generator import generate_candidates, _get_client


SCORING_PROMPT = """You are evaluating a proposed web navigation action.
Given the task, action history, the proposed action, and the predicted
page state after the action, rate how likely this action advances toward
completing the task.

Reply with EXACTLY two lines:
Reasoning: <brief one-sentence explanation>
Score: <float between 0.0 and 1.0>"""


def _action_to_description(action: dict, accessibility_tree: str) -> str:
    """Convert a structured action dict to a natural language description for Dreamer."""
    atype = action["action_type"]
    eid = action.get("element_id")
    value = action.get("value")

    if atype in ("scroll down", "scroll up", "go_back", "stop"):
        return atype

    # Look up element name from the accessibility tree
    element_desc = ""
    if eid is not None:
        for line in accessibility_tree.split("\n"):
            m = re.match(rf'^\[{eid}\]\s+(\w+)\s+"([^"]*)"', line)
            if m:
                element_desc = f'{m.group(1)} "{m.group(2)}"'
                break
        if not element_desc:
            element_desc = f"element [{eid}]"

    if atype == "type" and value:
        return f'type "{value}" in {element_desc}'
    elif atype == "click":
        return f"click {element_desc}"
    elif atype == "hover":
        return f"hover over {element_desc}"
    elif atype == "press" and value:
        return f'press "{value}"'
    else:
        return action.get("raw", atype)


def _score_candidate(task: str, action_history: list[str],
                     action_desc: str, predicted_state: str) -> float:
    """Score a single candidate action using Claude."""
    history_str = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(action_history)
    ) if action_history else "  (none)"

    user_prompt = f"""Task: {task}

Action history:
{history_str}

Proposed action: {action_desc}

Predicted page state after this action:
{predicted_state}

Rate this action."""

    try:
        resp = _get_client().messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=128,
            system=SCORING_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = resp.content[0].text.strip()
        m = re.search(r'Score:\s*([\d.]+)', text)
        if m:
            score = float(m.group(1))
            # Extract reasoning for logging
            r = re.search(r'Reasoning:\s*(.+)', text)
            reasoning = r.group(1).strip() if r else ""
            print(f"    Score {score:.2f} — {reasoning}")
            return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"    Scoring error: {e}")
    return 0.0


def plan_best_action(
    screenshot_b64: str,
    task: str,
    action_history: list[str],
    accessibility_tree: str,
    dreamer,
    current_url: str,
) -> list[dict]:
    """Generate candidates, predict outcomes with Dreamer-7B, score with Claude.

    Returns candidates sorted by score (best first).
    """
    # Step 1: Generate 3-5 candidates from Claude
    candidates = generate_candidates(
        screenshot_b64, task, action_history, accessibility_tree, multi=True
    )

    if not candidates:
        return []

    # If only 1 candidate, skip the planning overhead
    if len(candidates) == 1:
        print("  Only 1 candidate — skipping planning.")
        return candidates

    print(f"  Planning: evaluating {len(candidates)} candidates...")

    # Step 2: Predict outcomes with Dreamer-7B for each candidate
    predictions = []
    for i, cand in enumerate(candidates):
        desc = _action_to_description(cand, accessibility_tree)
        cand["_description"] = desc
        print(f"  [{i+1}/{len(candidates)}] Predicting: {desc}")

        if cand["action_type"] in ("scroll down", "scroll up"):
            pred = "The page scrolls to reveal more content."
        elif cand["action_type"] == "stop":
            pred = "The agent stops. The task ends with the current page state."
        else:
            try:
                pred = dreamer.state_change_prediction_in_website(
                    screenshot_b64, task, desc, format="change"
                )
            except Exception as e:
                print(f"    Dreamer error: {e}")
                pred = "(prediction failed)"
        predictions.append(pred)
        print(f"    Predicted: {pred[:120]}...")

    # Step 3: Score each candidate with Claude (in parallel)
    def score_fn(idx):
        return _score_candidate(
            task, action_history,
            candidates[idx]["_description"],
            predictions[idx],
        )

    with ThreadPoolExecutor(max_workers=5) as pool:
        scores = list(pool.map(score_fn, range(len(candidates))))

    # Attach scores and sort
    for cand, score in zip(candidates, scores):
        cand["_score"] = score

    candidates.sort(key=lambda c: c["_score"], reverse=True)

    print(f"  Planning result (best → worst):")
    for c in candidates:
        print(f"    {c['_score']:.2f}  {c['raw']}")

    return candidates
