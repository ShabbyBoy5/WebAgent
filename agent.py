"""
WebDreamer Agent — uses Claude to navigate real webpages via Playwright.

Usage:
    ~/webdreamer-env/bin/python agent.py \
        --task "Search for red blankets on Amazon" \
        --url "https://www.amazon.com" \
        --max-steps 15
"""

import argparse
import time

from browser_executor import BrowserManager
from action_generator import generate_candidates
from planning import plan_best_action


def _is_duplicate(action: dict, history: list[str]) -> bool:
    """Check if this action was already taken recently (loop detection)."""
    raw = action.get("raw", "").strip().lower()
    for prev in history[-3:]:
        if prev.strip().lower() == raw:
            return True
    return False


def run_agent(task: str, start_url: str, max_steps: int = 15,
              headless: bool = False, use_planning: bool = False):
    print(f"Task: {task}")
    print(f"URL:  {start_url}")
    print(f"Max steps: {max_steps}")
    print(f"Planning: {'ON' if use_planning else 'OFF'}")
    print()

    dreamer = None
    if use_planning:
        print("Loading Dreamer-7B world model...")
        from dreamer_model import DreamerWorldModel
        dreamer = DreamerWorldModel()
        print()

    browser = BrowserManager(headless=headless)
    browser.goto(start_url)

    action_history = []
    consecutive_failures = 0

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

        if use_planning:
            candidates = plan_best_action(
                screenshot_b64, task, action_history, tree_str,
                dreamer, browser.current_url(),
            )
        else:
            candidates = generate_candidates(
                screenshot_b64, task, action_history, tree_str
            )

        if not candidates:
            print("No valid actions generated. Retrying...")
            candidates = generate_candidates(
                screenshot_b64, task, action_history, tree_str
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
        print(f"Executing: {best['raw']}")

        if best["action_type"] == "stop":
            answer = best.get("value", "")
            print(f"\nAgent stopped. Answer: {answer}" if answer else "\nAgent stopped (task complete).")
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
        print()
    else:
        print(f"\nMax steps ({max_steps}) reached.")

    print(f"\nAction history:")
    for i, a in enumerate(action_history, 1):
        print(f"  {i}. {a}")

    browser.close()
    if dreamer is not None:
        del dreamer
    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebDreamer Agent")
    parser.add_argument("--task", required=True, help="Task to accomplish")
    parser.add_argument("--url", required=True, help="Starting URL")
    parser.add_argument("--max-steps", type=int, default=15, help="Max agent steps")
    parser.add_argument("--headless", action="store_true", help="Run browser headless")
    parser.add_argument("--planning", action="store_true", help="Enable Dreamer-7B planning pipeline")
    args = parser.parse_args()

    run_agent(args.task, args.url, args.max_steps, args.headless, args.planning)
