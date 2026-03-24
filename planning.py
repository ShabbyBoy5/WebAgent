"""
Planning pipeline: generate candidates -> predict outcomes with Dreamer-7B ->
[sentinel filter] -> score with a selectable dashboard scorer.
"""

import re
from concurrent.futures import ThreadPoolExecutor

from action_generator import generate_candidates, generate_full_plans
from llm_call import (
    call_text_llm,
    SCORING_MODEL_CLAUDE_OPUS,
    SCORING_MODEL_CODEX_SUBSCRIPTION,
)


PLAN_SCORING_PROMPT = """You are evaluating a complete web navigation plan.
Given the task, the starting page's interactive elements, and a proposed
multi-step plan, rate how likely this plan will successfully complete the task.

IMPORTANT context about how the executor works:
- "type [id] text" automatically presses Enter after typing. This means typing in a search box submits the search. There is NO need for a separate click on a search button — do NOT penalize plans for missing a search button click.
- Placeholder IDs like [?first_result] or [?search_box] are VALID. The executor resolves them at runtime by matching the description to real elements on the loaded page. Do NOT penalize plans for using placeholders — they are expected for future pages the plan hasn't seen yet.
- Only the first step needs to reference a real [id] from the current page. All later steps can and should use placeholders.

Consider:
- Does the plan cover all necessary steps to complete the task?
- Are the actions logical and in the right order?
- Does the first action correctly reference an element on the current page?
- Is the plan efficient?

Reply with EXACTLY two lines:
Reasoning: <brief explanation of strengths/weaknesses>
Score: <float between 0.0 and 1.0>"""


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
                     action_desc: str, predicted_state: str,
                     scoring_model: str = SCORING_MODEL_CLAUDE_OPUS) -> float:
    """Score a single candidate action using the configured scorer."""
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
        text = call_text_llm(
            SCORING_PROMPT,
            user_prompt,
            max_tokens=128,
            provider=scoring_model,
        )
        m = re.search(r'Score:\s*([\d.]+)', text)
        if m:
            score = float(m.group(1))
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
    sentinel_enabled: bool = False,
    corrective_rules: list[str] | None = None,
    session_context: str = "",
) -> list[dict]:
    """Generate candidates, predict outcomes with Dreamer-7B, score with Claude.

    Returns candidates sorted by score (best first).
    """
    # Step 1: Generate 3-5 candidates from Claude
    candidates = generate_candidates(
        screenshot_b64, task, action_history, accessibility_tree,
        multi=True, session_context=session_context,
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
        cand["_prediction"] = pred
        print(f"    Predicted: {pred[:120]}...")

    # Step 2.5: Sentinel safety filter (if enabled)
    if sentinel_enabled:
        from sentinel import filter_unsafe_candidates
        candidates, predictions, verdicts = filter_unsafe_candidates(
            candidates, predictions, current_url, corrective_rules
        )
        if not candidates:
            print("  SENTINEL: All candidates blocked as unsafe!")
            return []

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


def _score_plan(task: str, accessibility_tree: str, plan: list[dict],
                dreamer_prediction: str | None = None,
                plan_index: int = 0, event_callback=None,
                scoring_model: str = SCORING_MODEL_CLAUDE_OPUS) -> float:
    """Score a complete action plan using the configured scorer."""
    steps_str = "\n".join(
        f"  {i+1}. {a['raw']}" for i, a in enumerate(plan)
    )

    tree_display = accessibility_tree
    if len(tree_display) > 3000:
        tree_display = tree_display[:3000] + "\n... (truncated)"

    prediction_block = ""
    if dreamer_prediction:
        prediction_block = f"\nDreamer world-model prediction for step 1:\n{dreamer_prediction}\n"

    user_prompt = f"""Task: {task}

Current page interactive elements:
{tree_display}

Proposed plan:
{steps_str}
{prediction_block}
Rate this plan."""

    try:
        text = call_text_llm(
            PLAN_SCORING_PROMPT,
            user_prompt,
            max_tokens=256,
            provider=scoring_model,
        )
        m = re.search(r'Score:\s*([\d.]+)', text)
        if m:
            score = float(m.group(1))
            r = re.search(r'Reasoning:\s*(.+)', text)
            reasoning = r.group(1).strip() if r else ""
            print(f"    Score {score:.2f} — {reasoning}")
            if event_callback:
                event_callback("plan_score", {
                    "plan_index": plan_index,
                    "score": min(max(score, 0.0), 1.0),
                    "reasoning": reasoning,
                })
            return min(max(score, 0.0), 1.0)
    except Exception as e:
        print(f"    Plan scoring error: {e}")
    return 0.0


def select_best_plan(
    screenshot_b64: str,
    task: str,
    accessibility_tree: str,
    dreamer=None,
    current_url: str = "",
    num_plans: int = 3,
    min_steps: int = 0,
    scoring_model: str = SCORING_MODEL_CLAUDE_OPUS,
    session_context: str = "",
    event_callback=None,
) -> list[dict]:
    """Generate multiple complete plans, score them, return the best one.

    Uses Dreamer to predict the first step's outcome for each plan,
    and Sentinel to filter unsafe plans.

    Returns the best plan as a list of action dicts (ordered steps).
    """
    # Step 1: Generate N complete plans
    plans = generate_full_plans(
        screenshot_b64, task, accessibility_tree,
        num_plans=num_plans, min_steps=min_steps, session_context=session_context,
        event_callback=event_callback,
    )

    if not plans:
        return []

    if len(plans) == 1:
        print("  Only 1 plan generated — using it directly.")
        return plans[0]

    # Step 2: Dreamer predictions for each plan's first step
    if event_callback:
        event_callback("status", {"message": "Running world model predictions..."})
    dreamer_predictions = []
    for i, plan in enumerate(plans):
        first_action = plan[0]
        desc = _action_to_description(first_action, accessibility_tree)
        pred = None
        if dreamer and first_action["action_type"] not in ("scroll down", "scroll up", "stop"):
            try:
                print(f"  [Plan {i+1}] Dreamer predicting step 1: {desc}")
                pred = dreamer.state_change_prediction_in_website(
                    screenshot_b64, task, desc, format="change"
                )
                print(f"    Predicted: {pred[:120]}...")
                if event_callback:
                    event_callback("dreamer_predict", {
                        "plan_index": i,
                        "prediction": pred[:200],
                    })
            except Exception as e:
                print(f"    Dreamer error: {e}")
        dreamer_predictions.append(pred)

    # Step 3: Sentinel safety filter on first actions
    from sentinel import evaluate_candidate
    safe_plans = []
    safe_predictions = []
    safe_original_indices = []
    for i, (plan, pred) in enumerate(zip(plans, dreamer_predictions)):
        first_desc = _action_to_description(plan[0], accessibility_tree)
        verdict = evaluate_candidate(first_desc, pred, current_url)
        if verdict.is_safe:
            safe_plans.append(plan)
            safe_predictions.append(pred)
            safe_original_indices.append(i)
        else:
            print(f"  SENTINEL blocked Plan {i+1}: {verdict.risk_type} — {verdict.explanation}")
            if event_callback:
                event_callback("sentinel_block", {
                    "plan_index": i,
                    "risk_type": verdict.risk_type,
                    "explanation": verdict.explanation,
                })

    if not safe_plans:
        print("  SENTINEL: All plans blocked! Falling back to all plans.")
        safe_plans = plans
        safe_predictions = dreamer_predictions
        safe_original_indices = list(range(len(plans)))

    # Step 4: Score each safe plan (in parallel)
    if event_callback:
        event_callback("status", {"message": f"Scoring {len(safe_plans)} plans..."})
    print(f"  Scoring {len(safe_plans)} plans...")

    def score_fn(idx):
        print(f"  [Plan {idx+1}] Scoring ({len(safe_plans[idx])} steps)...")
        return _score_plan(
            task, accessibility_tree, safe_plans[idx], safe_predictions[idx],
            plan_index=safe_original_indices[idx],
            event_callback=event_callback,
            scoring_model=scoring_model,
        )

    max_workers = 1 if scoring_model == SCORING_MODEL_CODEX_SUBSCRIPTION else 5
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        scores = list(pool.map(score_fn, range(len(safe_plans))))

    # Step 5: Pick the best
    best_idx = max(range(len(safe_plans)), key=lambda i: scores[i])
    print(f"\n  Plan scores:")
    for i, (plan, score) in enumerate(zip(safe_plans, scores)):
        marker = " <-- BEST" if i == best_idx else ""
        steps_preview = " -> ".join(a["raw"] for a in plan[:4])
        if len(plan) > 4:
            steps_preview += f" -> ... ({len(plan)} steps total)"
        print(f"    Plan {i+1}: {score:.2f}  {steps_preview}{marker}")

    if event_callback:
        event_callback("best_plan", {
            "plan_index": safe_original_indices[best_idx],
            "score": scores[best_idx],
        })

    best_plan = safe_plans[best_idx]

    # Step 6: Predict ALL steps of the best plan with Dreamer (for Reflexion during execution)
    if dreamer:
        if event_callback:
            event_callback("status", {"message": "Predicting all plan steps with Dreamer..."})
        print(f"  Predicting all {len(best_plan)} steps of best plan...")
        for j, action in enumerate(best_plan):
            if action["action_type"] in ("scroll down", "scroll up", "stop"):
                continue
            desc = _action_to_description(action, accessibility_tree)
            try:
                pred = dreamer.state_change_prediction_in_website(
                    screenshot_b64, task, desc, format="change"
                )
                action["_prediction"] = pred
                action["_description"] = desc
                print(f"    Step {j+1}: {pred[:100]}...")
            except Exception as e:
                print(f"    Step {j+1} prediction error: {e}")

    return best_plan
