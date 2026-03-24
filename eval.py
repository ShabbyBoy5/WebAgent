"""
Evaluation harness — compare planning vs reactive mode on a task suite.

Usage:
    python eval.py                          # run all tasks, both modes
    python eval.py --mode plan-first        # only plan-first
    python eval.py --mode reactive          # only reactive
    python eval.py --tasks shopping         # only shopping category
    python eval.py --repeat 3              # repeat each task 3 times
    python eval.py --output results.json    # save results to file
"""

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass, field, asdict

# Unset proxy for Playwright / HuggingFace
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
if _proxy:
    os.environ["_SAVED_HTTPS_PROXY"] = _proxy
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

from browser_executor import BrowserManager
from action_generator import generate_candidates, generate_full_plans
from planning import select_best_plan, _action_to_description
from agent import _dreamer_reactive_step
from sentinel import evaluate_candidate


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

EVAL_TASKS = [
    # --- Search ---
    {
        "id": "search_1",
        "category": "search",
        "task": "Search for 'python web scraping' on DuckDuckGo",
        "url": "https://duckduckgo.com",
        "success_url_contains": "q=",
        "success_tree_contains": ["python", "scraping"],
    },
    {
        "id": "search_2",
        "category": "search",
        "task": "Search for 'best mechanical keyboards 2024' on Google",
        "url": "https://www.google.com",
        "success_url_contains": "q=",
        "success_tree_contains": ["keyboard"],
    },
    # --- Shopping ---
    {
        "id": "shopping_1",
        "category": "shopping",
        "task": "Find Sony headphones on Amazon",
        "url": "https://duckduckgo.com",
        "success_url_contains": "amazon",
        "success_tree_contains": ["sony", "headphone"],
    },
    {
        "id": "shopping_2",
        "category": "shopping",
        "task": "Search for a red blanket on Amazon",
        "url": "https://duckduckgo.com",
        "success_url_contains": "amazon",
        "success_tree_contains": ["blanket"],
    },
    # --- Navigation ---
    {
        "id": "nav_1",
        "category": "navigation",
        "task": "Go to Wikipedia and search for 'artificial intelligence'",
        "url": "https://duckduckgo.com",
        "success_url_contains": "wikipedia",
        "success_tree_contains": ["artificial intelligence"],
    },
    {
        "id": "nav_2",
        "category": "navigation",
        "task": "Go to GitHub and search for 'playwright python'",
        "url": "https://duckduckgo.com",
        "success_url_contains": "github",
        "success_tree_contains": ["playwright"],
    },
    # --- Multi-step ---
    {
        "id": "multi_1",
        "category": "multi_step",
        "task": "Search for 'laptop' on Amazon and sort by price low to high",
        "url": "https://duckduckgo.com",
        "success_url_contains": "amazon",
        "success_tree_contains": ["laptop"],
    },
]


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class StepLog:
    step: int
    action: str
    status: str
    url: str
    elapsed_ms: int


@dataclass
class EvalResult:
    task_id: str
    task: str
    mode: str  # "plan-first" or "reactive"
    success: bool
    steps_taken: int
    total_time_s: float
    error: str | None = None
    final_url: str = ""
    plan_count: int = 0
    plan_scores: list[float] = field(default_factory=list)
    steps_log: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Success checker
# ---------------------------------------------------------------------------

def check_success(browser: BrowserManager, task_def: dict) -> bool:
    """Check if the agent achieved the task goal."""
    url = browser.current_url().lower()
    tree, _ = browser.get_accessibility_tree()
    tree_lower = tree.lower()

    url_ok = True
    if task_def.get("success_url_contains"):
        url_ok = task_def["success_url_contains"].lower() in url

    tree_ok = True
    if task_def.get("success_tree_contains"):
        tree_ok = any(kw.lower() in tree_lower for kw in task_def["success_tree_contains"])

    return url_ok and tree_ok


# ---------------------------------------------------------------------------
# Run single eval
# ---------------------------------------------------------------------------

def run_eval_reactive(task_def: dict, max_steps: int = 15) -> EvalResult:
    """Run a single task in reactive mode."""
    browser = None
    try:
        browser = BrowserManager(headless=True, browser_type="firefox")
        browser.goto(task_def["url"])

        action_history = []
        steps_log = []
        t0 = time.time()

        for step in range(max_steps):
            screenshot_b64 = browser.screenshot_b64()
            tree_str, id_map = browser.get_accessibility_tree()

            step_t0 = time.time()
            candidates = generate_candidates(screenshot_b64, task_def["task"], action_history, tree_str)
            if not candidates:
                candidates = generate_candidates(screenshot_b64, task_def["task"], action_history, tree_str)
            if not candidates:
                steps_log.append({"step": step, "action": "(none)", "status": "no_candidates", "url": browser.current_url(), "elapsed_ms": int((time.time()-step_t0)*1000)})
                continue

            best = candidates[0]

            if best["action_type"] == "stop":
                steps_log.append({"step": step, "action": best["raw"], "status": "stopped", "url": browser.current_url(), "elapsed_ms": int((time.time()-step_t0)*1000)})
                break

            status = browser.execute(best["action_type"], best.get("element_id"), best.get("value"), id_map)
            elapsed = int((time.time() - step_t0) * 1000)
            steps_log.append({"step": step, "action": best["raw"], "status": status, "url": browser.current_url(), "elapsed_ms": elapsed})
            action_history.append(best["raw"])

            if status.startswith("error"):
                # try fallback
                for fb in candidates[1:]:
                    if fb["action_type"] == "stop":
                        continue
                    status = browser.execute(fb["action_type"], fb.get("element_id"), fb.get("value"), id_map)
                    if not status.startswith("error"):
                        action_history[-1] = fb["raw"]
                        break

        total_time = time.time() - t0
        success = check_success(browser, task_def)

        return EvalResult(
            task_id=task_def["id"],
            task=task_def["task"],
            mode="reactive",
            success=success,
            steps_taken=len(steps_log),
            total_time_s=round(total_time, 2),
            final_url=browser.current_url(),
            steps_log=steps_log,
        )

    except Exception as e:
        return EvalResult(
            task_id=task_def["id"],
            task=task_def["task"],
            mode="reactive",
            success=False,
            steps_taken=0,
            total_time_s=0,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass


def run_eval_reactive_dreamer(task_def: dict, max_steps: int = 15,
                              dreamer=None) -> EvalResult:
    """Run a single task in reactive mode with Dreamer proposing actions."""
    browser = None
    try:
        browser = BrowserManager(headless=True, browser_type="firefox")
        browser.goto(task_def["url"])

        action_history = []
        steps_log = []
        t0 = time.time()

        for step in range(max_steps):
            screenshot_b64 = browser.screenshot_b64()
            tree_str, id_map = browser.get_accessibility_tree()

            step_t0 = time.time()
            candidates = _dreamer_reactive_step(
                dreamer, screenshot_b64, task_def["task"], action_history, tree_str,
            )
            if not candidates:
                # Fallback to Claude if Dreamer fails
                candidates = generate_candidates(screenshot_b64, task_def["task"], action_history, tree_str)
            if not candidates:
                steps_log.append({"step": step, "action": "(none)", "status": "no_candidates", "url": browser.current_url(), "elapsed_ms": int((time.time()-step_t0)*1000)})
                continue

            best = candidates[0]

            if best["action_type"] == "stop":
                steps_log.append({"step": step, "action": best["raw"], "status": "stopped", "url": browser.current_url(), "elapsed_ms": int((time.time()-step_t0)*1000)})
                break

            status = browser.execute(best["action_type"], best.get("element_id"), best.get("value"), id_map)
            elapsed = int((time.time() - step_t0) * 1000)
            steps_log.append({"step": step, "action": best["raw"], "status": status, "url": browser.current_url(), "elapsed_ms": elapsed})
            action_history.append(best["raw"])

        total_time = time.time() - t0
        success = check_success(browser, task_def)

        return EvalResult(
            task_id=task_def["id"],
            task=task_def["task"],
            mode="reactive-dreamer",
            success=success,
            steps_taken=len(steps_log),
            total_time_s=round(total_time, 2),
            final_url=browser.current_url(),
            steps_log=steps_log,
        )

    except Exception as e:
        return EvalResult(
            task_id=task_def["id"],
            task=task_def["task"],
            mode="reactive-dreamer",
            success=False,
            steps_taken=0,
            total_time_s=0,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass


def run_eval_plan_first(task_def: dict, num_plans: int = 3, min_steps: int = 0,
                        dreamer=None) -> EvalResult:
    """Run a single task in plan-first mode."""
    browser = None
    try:
        browser = BrowserManager(headless=True, browser_type="firefox")
        browser.goto(task_def["url"])

        t0 = time.time()
        screenshot_b64 = browser.screenshot_b64()
        tree_str, id_map = browser.get_accessibility_tree()

        # Track scores via callback
        scores_collected = []
        def score_cb(event_type, data):
            if event_type == "plan_score":
                scores_collected.append(data.get("score", 0.0))

        plan = select_best_plan(
            screenshot_b64, task_def["task"], tree_str,
            dreamer=dreamer,
            current_url=browser.current_url(),
            num_plans=num_plans,
            min_steps=min_steps,
            event_callback=score_cb,
        )

        if not plan:
            return EvalResult(
                task_id=task_def["id"],
                task=task_def["task"],
                mode="plan-first",
                success=False,
                steps_taken=0,
                total_time_s=round(time.time() - t0, 2),
                error="No plans generated",
                plan_count=0,
            )

        # Execute plan
        steps_log = []
        for step_num, action in enumerate(plan):
            if action["action_type"] == "stop":
                steps_log.append({"step": step_num, "action": action["raw"], "status": "stopped", "url": browser.current_url(), "elapsed_ms": 0})
                break

            _, current_id_map = browser.get_accessibility_tree()
            step_t0 = time.time()
            status = browser.execute(
                action["action_type"],
                action.get("element_id"),
                action.get("value"),
                current_id_map,
            )
            elapsed = int((time.time() - step_t0) * 1000)
            steps_log.append({"step": step_num, "action": action["raw"], "status": status, "url": browser.current_url(), "elapsed_ms": elapsed})

            if status.startswith("error"):
                break

            time.sleep(0.5)

        total_time = time.time() - t0
        success = check_success(browser, task_def)

        return EvalResult(
            task_id=task_def["id"],
            task=task_def["task"],
            mode="plan-first",
            success=success,
            steps_taken=len(steps_log),
            total_time_s=round(total_time, 2),
            final_url=browser.current_url(),
            plan_count=num_plans,
            plan_scores=scores_collected,
            steps_log=steps_log,
        )

    except Exception as e:
        return EvalResult(
            task_id=task_def["id"],
            task=task_def["task"],
            mode="plan-first",
            success=False,
            steps_taken=0,
            total_time_s=0,
            error=f"{type(e).__name__}: {e}",
        )
    finally:
        if browser:
            try:
                browser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_evals(modes: list[str], categories: list[str] | None, num_plans: int,
              min_steps: int, max_steps: int, repeat: int,
              use_dreamer: bool = False) -> list[EvalResult]:
    """Run the full eval suite."""
    tasks = EVAL_TASKS
    if categories:
        tasks = [t for t in tasks if t["category"] in categories]

    # Load Dreamer once if requested
    dreamer = None
    if use_dreamer:
        print("Loading Dreamer-7B world model (one-time)...")
        from dreamer_model import DreamerWorldModel
        dreamer = DreamerWorldModel()
        print("Dreamer loaded.\n")

    results = []
    total = len(tasks) * len(modes) * repeat
    done = 0

    try:
        for task_def in tasks:
            for mode in modes:
                for run_idx in range(repeat):
                    done += 1
                    run_label = f"[{done}/{total}]"
                    print(f"\n{'='*70}")
                    print(f"{run_label} {task_def['id']} | {mode} | run {run_idx+1}/{repeat}")
                    print(f"  Task: {task_def['task']}")

                    if mode == "reactive-dreamer":
                        result = run_eval_reactive_dreamer(task_def, max_steps=max_steps, dreamer=dreamer)
                    elif mode == "plan-first":
                        result = run_eval_plan_first(task_def, num_plans=num_plans,
                                                     min_steps=min_steps, dreamer=dreamer)
                    else:
                        result = run_eval_reactive(task_def, max_steps=max_steps)

                    status = "PASS" if result.success else "FAIL"
                    print(f"  Result: {status} | {result.steps_taken} steps | {result.total_time_s}s")
                    if result.error:
                        print(f"  Error: {result.error}")
                    if result.plan_scores:
                        print(f"  Plan scores: {result.plan_scores}")
                    print(f"  Final URL: {result.final_url}")

                    results.append(result)
    finally:
        if dreamer is not None:
            del dreamer

    return results


def print_summary(results: list[EvalResult]):
    """Print a comparison table."""
    print(f"\n\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}\n")

    all_modes = list(dict.fromkeys(r.mode for r in results))
    for mode in all_modes:
        mode_results = [r for r in results if r.mode == mode]

        successes = sum(1 for r in mode_results if r.success)
        total = len(mode_results)
        avg_steps = sum(r.steps_taken for r in mode_results) / total if total else 0
        avg_time = sum(r.total_time_s for r in mode_results) / total if total else 0
        errors = sum(1 for r in mode_results if r.error)

        print(f"  {mode.upper()}")
        print(f"    Success rate:  {successes}/{total} ({100*successes/total:.0f}%)")
        print(f"    Avg steps:     {avg_steps:.1f}")
        print(f"    Avg time:      {avg_time:.1f}s")
        print(f"    Errors:        {errors}")
        print()

    # Per-task comparison
    header = f"  {'TASK':<20}" + "".join(f" {m.upper():<18}" for m in all_modes)
    print(header)
    print(f"  {'-'*(20 + 18*len(all_modes))}")

    task_ids = list(dict.fromkeys(r.task_id for r in results))
    for tid in task_ids:
        row = f"  {tid:<20}"
        for mode in all_modes:
            mr = [r for r in results if r.task_id == tid and r.mode == mode]
            rate = f"{sum(r.success for r in mr)}/{len(mr)}" if mr else "-"
            row += f" {rate:<18}"
        print(row)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebAgent Evaluation")
    parser.add_argument("--mode", choices=["reactive-dreamer", "plan-first", "both"], default="both",
                        help="Which mode(s) to evaluate (both = reactive-dreamer + plan-first)")
    parser.add_argument("--tasks", nargs="*", help="Task categories to run (search, shopping, navigation, multi_step)")
    parser.add_argument("--num-plans", type=int, default=3, help="Plans to generate in plan-first mode")
    parser.add_argument("--min-steps", type=int, default=0, help="Min steps per plan")
    parser.add_argument("--max-steps", type=int, default=15, help="Max steps in reactive mode")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each task N times")
    parser.add_argument("--dreamer", action="store_true", help="Use Dreamer-7B world model (loaded once, reused across all tasks)")
    parser.add_argument("--output", default="eval_results.json", help="Output file for results")
    args = parser.parse_args()

    modes = ["reactive-dreamer", "plan-first"] if args.mode == "both" else [args.mode]

    # Both modes need Dreamer — always load it
    results = run_evals(
        modes=modes,
        categories=args.tasks,
        num_plans=args.num_plans,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        repeat=args.repeat,
        use_dreamer=True,
    )

    print_summary(results)

    # Save results
    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "modes": modes,
            "categories": args.tasks,
            "num_plans": args.num_plans,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps,
            "repeat": args.repeat,
        },
        "results": [asdict(r) for r in results],
        "summary": {},
    }

    for mode in modes:
        mr = [r for r in results if r.mode == mode]
        if mr:
            output_data["summary"][mode] = {
                "success_rate": sum(r.success for r in mr) / len(mr),
                "avg_steps": sum(r.steps_taken for r in mr) / len(mr),
                "avg_time_s": sum(r.total_time_s for r in mr) / len(mr),
                "error_count": sum(1 for r in mr if r.error),
            }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.output}")
