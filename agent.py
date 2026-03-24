"""
WebDreamer Agent — uses Claude to navigate real webpages via Playwright.

Usage:
    python agent.py \
        --task "Search for red blankets on Amazon" \
        --url "https://www.amazon.com" \
        --max-steps 15 \
        --planning --sentinel
"""

import argparse
import os
import time

# Save proxy URL for token refresh, then unset for HuggingFace and Playwright
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
if _proxy:
    os.environ["_SAVED_HTTPS_PROXY"] = _proxy
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

from browser_executor import BrowserManager
from action_generator import generate_candidates
from planning import plan_best_action, select_best_plan
from session_memory import SessionMemory
from reflexion import compare_states
from sentinel import evaluate_candidate


def _dreamer_reactive_step(dreamer, screenshot_b64, task, action_history, tree_str):
    """Use Dreamer to propose a single action, then convert to structured format."""
    from action_generator import _parse_action
    from llm_call import call_text_llm

    # Build imaginations list from history (action, prediction) pairs
    # For reactive mode we don't have predictions, so just pass action descriptions
    imaginations = [(a, "") for a in action_history]

    if imaginations:
        raw_action = dreamer.action_proposal_in_imagination(
            screenshot_b64, task, imaginations, format="change"
        )
    else:
        # First step — no history, ask Dreamer directly
        prompt = f"You are looking at a webpage. Your task: {task}\nBased on the screenshot, what is the single best next action? Describe it briefly (e.g. 'click on the search box', 'type sony headphones in the search field')."
        messages = [
            {"role": "system", "content": "You are an autonomous intelligent agent tasked with navigating a web browser. Output a single short action description."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": screenshot_b64}},
            ]},
        ]
        raw_action = dreamer._call_model(messages)

    print(f"  Dreamer proposes: {raw_action}")

    # Convert Dreamer's natural language action to structured format
    convert_prompt = f"""Convert this action description into the correct format.

Action: {raw_action}

Interactive elements on the page:
{tree_str}

Reply with EXACTLY one line in one of these formats:
type [<id>] "<text>"
click [<id>]
scroll down
scroll up
go_back
stop

Rules:
- type [id] "text" auto-presses Enter
- Match the action description to the best element from the list
- Output ONLY the action line, nothing else"""

    structured = call_text_llm(
        "Convert natural language web actions to structured format. Output ONLY the action line.",
        convert_prompt,
        max_tokens=64,
    )
    print(f"  Structured: {structured}")

    parsed = _parse_action(structured)
    if parsed:
        parsed["_description"] = raw_action
        return [parsed]
    return []


def _is_duplicate(action: dict, history: list[str]) -> bool:
    """Check if this action was already taken recently (loop detection)."""
    raw = action.get("raw", "").strip().lower()
    for prev in history[-3:]:
        if prev.strip().lower() == raw:
            return True
    return False


def run_agent(task: str, start_url: str, max_steps: int = 15,
              headless: bool = False, use_planning: bool = False,
              browser_type: str = "chromium", use_sentinel: bool = False,
              plan_first: bool = False, keep_open: bool = False,
              num_plans: int = 3, min_steps: int = 0,
              use_dreamer: bool = False):
    print(f"Task: {task}")
    print(f"URL:  {start_url}")
    print(f"Max steps: {max_steps}")
    mode = "PLAN-FIRST" if plan_first else ("PLANNING" if use_planning else "REACTIVE")
    print(f"Mode: {mode}")
    print(f"Sentinel: {'ON' if use_sentinel else 'OFF'}")
    print()

    dreamer = None
    if use_planning or plan_first or use_dreamer:
        print("Loading Dreamer-7B world model...")
        from dreamer_model import DreamerWorldModel
        dreamer = DreamerWorldModel()
        print()

    browser = BrowserManager(headless=headless, browser_type=browser_type)
    browser.goto(start_url)

    # --- PLAN-FIRST MODE ---
    if plan_first:
        screenshot_b64 = browser.screenshot_b64()
        tree_str, id_map = browser.get_accessibility_tree()
        session_context = ""

        print("Generating and scoring plans (with Dreamer)...")
        plan = select_best_plan(
            screenshot_b64, task, tree_str,
            dreamer=dreamer,
            current_url=browser.current_url(),
            num_plans=num_plans, min_steps=min_steps, session_context=session_context,
        )

        if not plan:
            print("No plans generated. Aborting.")
            if not keep_open:
                browser.close()
            if dreamer is not None:
                del dreamer
            return

        print(f"\nExecuting best plan ({len(plan)} steps)...\n")

        action_history = []
        memory = SessionMemory()

        for step_num, action in enumerate(plan, 1):
            print(f"{'='*60}")
            print(f"STEP {step_num}/{len(plan)}")
            print(f"URL: {browser.current_url()}")

            if action["action_type"] == "stop":
                print(f"Executing: {action['raw']}")
                print("\nAgent stopped (task complete).")
                memory.add_entry(step=step_num, action=action["raw"])
                break

            # Re-read state for this step
            screenshot_b64 = browser.screenshot_b64()
            current_tree, current_id_map = browser.get_accessibility_tree()

            # Dreamer: predict outcome
            from planning import _action_to_description
            desc = _action_to_description(action, current_tree)
            predicted_state = None
            if action["action_type"] not in ("scroll down", "scroll up"):
                try:
                    predicted_state = dreamer.state_change_prediction_in_website(
                        screenshot_b64, task, desc, format="change"
                    )
                    print(f"  Dreamer prediction: {predicted_state[:120]}...")
                except Exception as e:
                    print(f"  Dreamer error: {e}")

            # Sentinel: safety check before executing
            corrective_rules = memory.get_corrective_rules() or None
            verdict = evaluate_candidate(desc, predicted_state, browser.current_url(), corrective_rules)
            if not verdict.is_safe:
                print(f"  SENTINEL BLOCKED: {desc}")
                print(f"    Risk: {verdict.risk_type} — {verdict.explanation}")
                memory.add_entry(step=step_num, action=action["raw"], was_safe=False)
                print("  Skipping unsafe step, continuing plan...")
                continue

            print(f"Executing: {action['raw']}")

            status = browser.execute(
                action["action_type"],
                action.get("element_id"),
                action.get("value"),
                current_id_map,
            )
            print(f"Result: {status}")

            if status.startswith("error"):
                print(f"  Plan step failed — stopping early.")
                memory.add_entry(step=step_num, action=action["raw"],
                                 actual_summary=f"Status: {status}")
                break

            action_history.append(action["raw"])

            # Reflexion: compare predicted vs actual state
            reflexion_note = None
            if predicted_state and not status.startswith("error"):
                time.sleep(0.5)
                post_tree, _ = browser.get_accessibility_tree()
                if post_tree.strip():
                    print("  Running reflexion...")
                    result = compare_states(
                        action_taken=action["raw"],
                        predicted_state=predicted_state,
                        actual_tree=post_tree,
                        task=task,
                    )
                    print(f"  Reflexion: match={result.match_score:.2f} — {result.diagnosis}")
                    if result.mismatch_detected:
                        print(f"  MISMATCH! Rule: {result.corrective_rule}")
                        reflexion_note = result.corrective_rule

            memory.add_entry(
                step=step_num,
                action=action["raw"],
                predicted_state=predicted_state,
                actual_summary=f"Status: {status}",
                was_safe=True,
                reflexion_note=reflexion_note,
            )
            print()

        print(f"\nAction history:")
        for i, a in enumerate(action_history, 1):
            print(f"  {i}. {a}")

        rules = memory.get_corrective_rules()
        if rules:
            print(f"\nLearned rules from this session:")
            for i, rule in enumerate(rules, 1):
                print(f"  {i}. {rule}")

        if dreamer is not None:
            del dreamer

        if keep_open:
            print("\nBrowser left open. Press Enter to close...")
            input()

        browser.close()
        print("\nDone.")
        return

    # --- REACTIVE / STEP-BY-STEP MODE ---
    action_history = []
    consecutive_failures = 0
    memory = SessionMemory()

    for step in range(1, max_steps + 1):
        print(f"{'='*60}")
        print(f"STEP {step}/{max_steps}")
        print(f"URL: {browser.current_url()}")

        screenshot_b64 = browser.screenshot_b64()
        tree_str, id_map = browser.get_accessibility_tree()

        if not tree_str.strip():
            print("Warning: empty accessibility tree, waiting and retrying...")
            time.sleep(2)
            tree_str, id_map = browser.get_accessibility_tree()

        print(f"Interactive elements found: {len(id_map)}")

        # Build session context from memory
        session_context = memory.format_for_prompt()
        corrective_rules = memory.get_corrective_rules() or None

        if use_planning:
            candidates = plan_best_action(
                screenshot_b64, task, action_history, tree_str,
                dreamer, browser.current_url(),
                sentinel_enabled=use_sentinel,
                corrective_rules=corrective_rules,
                session_context=session_context,
            )
        elif use_dreamer and dreamer:
            # Reactive with Dreamer: Dreamer proposes action, convert to structured format
            candidates = _dreamer_reactive_step(
                dreamer, screenshot_b64, task, action_history, tree_str,
            )
        else:
            candidates = generate_candidates(
                screenshot_b64, task, action_history, tree_str,
                session_context=session_context,
            )

        if not candidates:
            print("No valid actions generated. Retrying...")
            candidates = generate_candidates(
                screenshot_b64, task, action_history, tree_str,
                session_context=session_context,
            )

        if not candidates:
            consecutive_failures += 1
            if consecutive_failures >= 3:
                print("Too many failures. Stopping.")
                break
            print("No actions parsed. Skipping step.")
            continue

        # Filter duplicates
        filtered = [c for c in candidates if not _is_duplicate(c, action_history)]
        if not filtered:
            print("All candidates are duplicates. Scrolling as fallback.")
            filtered = [{"action_type": "scroll down", "element_id": None, "value": None, "raw": "scroll down"}]

        best = filtered[0]

        # Sentinel check in reactive mode (planning mode handles it internally)
        if use_sentinel and not use_planning:
            desc = best.get("_description", best.get("raw", ""))
            predicted = best.get("_prediction")
            verdict = evaluate_candidate(desc, predicted, browser.current_url(), corrective_rules)
            if not verdict.is_safe:
                print(f"SENTINEL BLOCKED: {desc}")
                print(f"  Risk: {verdict.risk_type} — {verdict.explanation}")
                # Try fallbacks
                safe_found = False
                for fallback in filtered[1:]:
                    fb_desc = fallback.get("_description", fallback.get("raw", ""))
                    fb_verdict = evaluate_candidate(fb_desc, None, browser.current_url(), corrective_rules)
                    if fb_verdict.is_safe:
                        best = fallback
                        safe_found = True
                        break
                    else:
                        print(f"  SENTINEL also blocked fallback: {fb_desc}")
                if not safe_found:
                    print("  All candidates blocked by Sentinel. Skipping step.")
                    memory.add_entry(step=step, action=best["raw"], was_safe=False)
                    continue

        print(f"Executing: {best['raw']}")

        if best["action_type"] == "stop":
            answer = best.get("value", "")
            print(f"\nAgent stopped. Answer: {answer}" if answer else "\nAgent stopped (task complete).")
            memory.add_entry(step=step, action=best["raw"])
            break

        status = browser.execute(
            best["action_type"],
            best.get("element_id"),
            best.get("value"),
            id_map,
        )
        print(f"Result: {status}")

        if status.startswith("error"):
            consecutive_failures += 1
            for fallback in filtered[1:]:
                if fallback["action_type"] == "stop":
                    continue
                print(f"  Trying fallback: {fallback['raw']}")
                status = browser.execute(
                    fallback["action_type"],
                    fallback.get("element_id"),
                    fallback.get("value"),
                    id_map,
                )
                print(f"  Result: {status}")
                if not status.startswith("error"):
                    best = fallback
                    consecutive_failures = 0
                    break
        else:
            consecutive_failures = 0

        action_history.append(best["raw"])

        # Post-execution: Reflexion (only when planning provided a prediction)
        predicted_state = best.get("_prediction")
        reflexion_note = None
        if use_planning and predicted_state and not status.startswith("error"):
            time.sleep(0.5)  # let page settle
            post_tree, _ = browser.get_accessibility_tree()
            if post_tree.strip():
                print("  Running reflexion...")
                result = compare_states(
                    action_taken=best["raw"],
                    predicted_state=predicted_state,
                    actual_tree=post_tree,
                    task=task,
                )
                print(f"  Reflexion: match={result.match_score:.2f} — {result.diagnosis}")
                if result.mismatch_detected:
                    print(f"  MISMATCH! Rule: {result.corrective_rule}")
                    reflexion_note = result.corrective_rule

        # Record step in session memory
        memory.add_entry(
            step=step,
            action=best["raw"],
            predicted_state=predicted_state,
            actual_summary=f"Status: {status}",
            was_safe=True,
            reflexion_note=reflexion_note,
        )

        print()
    else:
        print(f"\nMax steps ({max_steps}) reached.")

    print(f"\nAction history:")
    for i, a in enumerate(action_history, 1):
        print(f"  {i}. {a}")

    # Print learned rules if any
    rules = memory.get_corrective_rules()
    if rules:
        print(f"\nLearned rules from this session:")
        for i, rule in enumerate(rules, 1):
            print(f"  {i}. {rule}")

    if dreamer is not None:
        del dreamer

    if keep_open:
        print("\nBrowser left open. Press Enter to close...")
        input()

    browser.close()
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebDreamer Agent")
    parser.add_argument("--task", required=True, help="Task to accomplish")
    parser.add_argument("--url", default="https://duckduckgo.com", help="Starting URL (default: DuckDuckGo)")
    parser.add_argument("--max-steps", type=int, default=15, help="Max agent steps")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--browser", default="firefox", choices=["chromium", "firefox", "webkit"],
                        help="Browser engine (firefox for Zen-like experience)")
    parser.add_argument("--planning", action="store_true", help="Enable Dreamer-7B planning pipeline")
    parser.add_argument("--sentinel", action="store_true", help="Enable Sentinel safety filtering")
    parser.add_argument("--plan-first", action="store_true",
                        help="Plan all steps upfront, score plans, execute best one in one shot")
    parser.add_argument("--keep-open", action="store_true",
                        help="Keep browser open after agent finishes (press Enter to close)")
    parser.add_argument("--num-plans", type=int, default=3,
                        help="Number of plans to generate in plan-first mode (default: 3)")
    parser.add_argument("--min-steps", type=int, default=0,
                        help="Minimum number of action steps each plan must have (default: 0 = no minimum)")
    parser.add_argument("--dreamer", action="store_true",
                        help="Use Dreamer-7B for reactive action proposals (instead of Claude)")
    args = parser.parse_args()

    run_agent(args.task, args.url, args.max_steps, args.headless, args.planning,
              args.browser, args.sentinel, args.plan_first, args.keep_open,
              args.num_plans, args.min_steps, args.dreamer)
