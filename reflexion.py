"""
Reflexion module — compares predicted vs actual state after action execution
and generates corrective rules when mismatches are detected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from llm_call import call_text_llm


MISMATCH_THRESHOLD = 0.6


@dataclass
class ReflexionResult:
    match_score: float        # 0.0-1.0
    mismatch_detected: bool   # True if score < threshold
    diagnosis: str
    corrective_rule: str | None


REFLEXION_SYSTEM = """You are analyzing whether a predicted web page state matches the actual observed state after an action was taken.

You will receive:
1. The action that was taken
2. The predicted state (from a world model)
3. The actual state (accessibility tree of the real page after the action)
4. The task the agent is trying to accomplish

Evaluate the match and diagnose any discrepancy.

Reply with EXACTLY three lines:
Match: <float between 0.0 and 1.0> (1.0 = perfect match, 0.0 = completely wrong)
Diagnosis: <one sentence explaining match or mismatch>
Rule: <corrective rule for future actions, or NONE if match >= 0.7>

Example (mismatch):
Match: 0.2
Diagnosis: The prediction expected search results but the page showed a CAPTCHA challenge instead.
Rule: On this site, rapid navigation triggers CAPTCHAs — slow down and verify page state before proceeding.

Example (good match):
Match: 0.9
Diagnosis: The search results appeared as predicted with minor layout differences.
Rule: NONE"""


def compare_states(
    action_taken: str,
    predicted_state: str,
    actual_tree: str,
    task: str,
) -> ReflexionResult:
    """Compare predicted state with actual observed state.

    Args:
        action_taken: The action that was executed
        predicted_state: Dreamer's prediction string
        actual_tree: Accessibility tree from the real page after execution
        task: The overall task being accomplished
    """
    actual_display = actual_tree[:3000] if len(actual_tree) > 3000 else actual_tree
    pred_display = predicted_state[:2000] if len(predicted_state) > 2000 else predicted_state

    user_prompt = f"""Task: {task}

Action taken: {action_taken}

Predicted state after action:
{pred_display}

Actual page state (accessibility tree):
{actual_display}

Evaluate the match between predicted and actual state."""

    try:
        text = call_text_llm(REFLEXION_SYSTEM, user_prompt)
        return _parse_reflexion(text)
    except Exception as e:
        print(f"  Reflexion error: {e}")
        return ReflexionResult(
            match_score=1.0,
            mismatch_detected=False,
            diagnosis=f"Reflexion error: {e}",
            corrective_rule=None,
        )


def _parse_reflexion(text: str) -> ReflexionResult:
    match_m = re.search(r'Match:\s*([\d.]+)', text)
    diag_m = re.search(r'Diagnosis:\s*(.+)', text)
    rule_m = re.search(r'Rule:\s*(.+)', text)

    match_score = float(match_m.group(1)) if match_m else 0.5
    match_score = min(max(match_score, 0.0), 1.0)

    diagnosis = diag_m.group(1).strip() if diag_m else "Could not parse diagnosis"

    rule_text = rule_m.group(1).strip() if rule_m else None
    corrective_rule = None
    if rule_text and rule_text.upper() != "NONE":
        corrective_rule = rule_text

    mismatch_detected = match_score < MISMATCH_THRESHOLD

    return ReflexionResult(
        match_score=match_score,
        mismatch_detected=mismatch_detected,
        diagnosis=diagnosis,
        corrective_rule=corrective_rule if mismatch_detected else None,
    )
