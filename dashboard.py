"""
WebAgent Dashboard — FastAPI backend with SSE streaming.

Usage:
    uv run uvicorn dashboard:app --port 8000
"""

import os
import json
import queue
import threading
import uuid
import asyncio
from dataclasses import dataclass, field

# Unset proxy for Playwright / HuggingFace downloads
_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
if _proxy:
    os.environ["_SAVED_HTTPS_PROXY"] = _proxy
for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
    os.environ.pop(var, None)

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from browser_executor import BrowserManager
from action_generator import generate_full_plans, generate_candidates, kill_active_llm_call as kill_action_llm
from agent import _dreamer_reactive_step
from llm_call import (
    kill_active_llm_call as kill_scoring_llm,
    SCORING_MODEL_CLAUDE_OPUS,
    SCORING_MODEL_CODEX_SUBSCRIPTION,
)
from planning import select_best_plan, _action_to_description
from sentinel import evaluate_candidate
from reflexion import compare_states
from session_memory import SessionMemory

import time

app = FastAPI(title="WebAgent Dashboard")

# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)


# --- Models ---

class RunRequest(BaseModel):
    task: str
    start_url: str = "https://duckduckgo.com"
    model: str = "claude"  # "dreamer" or "claude"
    scoring_model: str = SCORING_MODEL_CLAUDE_OPUS
    mode: str = "plan-first"  # "plan-first" or "reactive"
    num_plans: int = 3
    min_steps: int = 0
    max_steps: int = 15
    browser_type: str = "firefox"


@dataclass
class DashboardSession:
    session_id: str
    config: RunRequest
    event_queue: queue.Queue = field(default_factory=queue.Queue)
    thread: threading.Thread | None = None
    browser: BrowserManager | None = None
    cancelled: bool = False
    status: str = "starting"


# Single active session
_active_session: DashboardSession | None = None
_session_lock = threading.Lock()

# Persistent memory across runs — lives until dashboard restarts
_persistent_memory = SessionMemory()


def _emit(session: DashboardSession, event_type: str, data: dict | None = None):
    """Push a structured event to the session's queue."""
    event = {"type": event_type}
    if data:
        event.update(data)
    session.event_queue.put(event)


class CancelledError(Exception):
    pass


def _event_callback(session: DashboardSession):
    """Return a callback function for passing to generate_full_plans / select_best_plan.

    Also checks the cancel flag on every callback — raises CancelledError to
    unwind the call stack immediately instead of waiting for the next phase check.
    """
    def cb(event_type: str, data: dict):
        if session.cancelled:
            raise CancelledError("Cancelled by user")
        _emit(session, event_type, data)
    return cb


# --- Agent Orchestration (runs in background thread) ---

def _execute_step(session, step_num, action, total, config, task=""):
    """Execute a single action step. Returns True to continue, False to stop."""
    _emit(session, "exec_step", {
        "step": step_num,
        "total": total,
        "raw": action["raw"],
        "action_type": action["action_type"],
        "status": "running",
    })

    if action["action_type"] == "stop":
        _emit(session, "exec_result", {"step": step_num, "status": "stopped"})
        _persistent_memory.add_entry(step=step_num, action=action["raw"])
        return False

    current_tree, current_id_map = session.browser.get_accessibility_tree()

    # Sentinel check (with learned rules from previous runs)
    corrective_rules = _persistent_memory.get_corrective_rules() or None
    desc = _action_to_description(action, current_tree)
    verdict = evaluate_candidate(desc, None, session.browser.current_url(), corrective_rules)

    # Always emit sentinel verdict (SAFE or UNSAFE)
    _emit(session, "sentinel_verdict", {
        "step": step_num,
        "is_safe": verdict.is_safe,
        "risk_type": verdict.risk_type,
        "explanation": verdict.explanation,
    })

    if not verdict.is_safe:
        _emit(session, "exec_result", {"step": step_num, "status": "blocked"})
        _persistent_memory.add_entry(step=step_num, action=action["raw"], was_safe=False)
        return True  # skip but continue

    status = session.browser.execute(
        action["action_type"],
        action.get("element_id"),
        action.get("value"),
        current_id_map,
    )

    _emit(session, "exec_result", {"step": step_num, "status": status})

    if status.startswith("error"):
        _persistent_memory.add_entry(step=step_num, action=action["raw"],
                                     actual_summary=f"Status: {status}")
        return False

    time.sleep(0.5)
    screenshot_b64 = session.browser.screenshot_b64()
    _emit(session, "screenshot", {"data": screenshot_b64})

    # Reflexion: compare predicted vs actual if prediction available
    predicted = action.get("_prediction")
    reflexion_note = None
    if predicted:
        post_tree, _ = session.browser.get_accessibility_tree()
        if post_tree.strip():
            result = compare_states(
                action_taken=action["raw"],
                predicted_state=predicted,
                actual_tree=post_tree,
                task=task,
            )
            _emit(session, "reflexion", {
                "step": step_num,
                "match_score": result.match_score,
                "diagnosis": result.diagnosis,
                "mismatch": result.mismatch_detected,
            })
            if result.mismatch_detected:
                reflexion_note = result.corrective_rule

    _persistent_memory.add_entry(
        step=step_num,
        action=action["raw"],
        predicted_state=predicted,
        actual_summary=f"Status: {status}",
        was_safe=True,
        reflexion_note=reflexion_note,
    )
    return True


def run_dashboard_agent(session: DashboardSession):
    """Agent orchestrator with SSE event streaming. Supports plan-first and reactive modes."""
    global _active_session
    config = session.config
    cb = _event_callback(session)

    try:
        # Phase 1: Setup browser
        _emit(session, "status", {"phase": "starting", "message": "Launching browser..."})
        session.browser = BrowserManager(headless=True, browser_type=config.browser_type)
        session.browser.goto(config.start_url)

        screenshot_b64 = session.browser.screenshot_b64()
        tree_str, id_map = session.browser.get_accessibility_tree()
        _emit(session, "screenshot", {"data": screenshot_b64})

        if session.cancelled:
            raise InterruptedError("Cancelled")

        # --- REACTIVE MODE ---
        if config.mode == "reactive":
            # Load Dreamer if selected
            dreamer = None
            if config.model == "dreamer":
                _emit(session, "status", {"phase": "loading_model", "message": "Loading Dreamer-7B..."})
                from dreamer_model import DreamerWorldModel
                dreamer = DreamerWorldModel()

            _emit(session, "status", {"phase": "executing", "message": "Running in reactive mode..."})
            _emit(session, "execution_start", {"total_steps": config.max_steps})

            action_history = []
            for step in range(config.max_steps):
                if session.cancelled:
                    raise InterruptedError("Cancelled")

                screenshot_b64 = session.browser.screenshot_b64()
                tree_str, id_map = session.browser.get_accessibility_tree()
                _emit(session, "screenshot", {"data": screenshot_b64})

                _emit(session, "status", {"phase": "executing", "message": f"Thinking (step {step+1})..."})

                session_context = _persistent_memory.format_for_prompt()
                if dreamer:
                    candidates = _dreamer_reactive_step(
                        dreamer, screenshot_b64, config.task, action_history, tree_str,
                    )
                else:
                    candidates = generate_candidates(
                        screenshot_b64, config.task, action_history, tree_str,
                        session_context=session_context,
                    )
                if not candidates:
                    _emit(session, "status", {"phase": "executing", "message": f"Retrying step {step+1}..."})
                    candidates = generate_candidates(
                        screenshot_b64, config.task, action_history, tree_str,
                    )
                if not candidates:
                    _emit(session, "exec_step", {
                        "step": step, "total": config.max_steps,
                        "raw": "(no action generated)", "action_type": "error", "status": "error",
                    })
                    continue

                best = candidates[0]
                keep_going = _execute_step(session, step, best, config.max_steps, config, task=config.task)
                action_history.append(best["raw"])
                if not keep_going:
                    break

            if dreamer is not None:
                del dreamer

        # --- PLAN-FIRST MODE ---
        else:
            _emit(session, "status", {"phase": "planning", "message": "Generating plans..."})

            if config.scoring_model not in (SCORING_MODEL_CLAUDE_OPUS, SCORING_MODEL_CODEX_SUBSCRIPTION):
                raise ValueError(f"Unsupported scoring model: {config.scoring_model}")

            dreamer = None
            if config.model == "dreamer":
                _emit(session, "status", {"phase": "loading_model", "message": "Loading Dreamer-7B world model..."})
                from dreamer_model import DreamerWorldModel
                dreamer = DreamerWorldModel()

            session_context = _persistent_memory.format_for_prompt()
            plan = select_best_plan(
                screenshot_b64, config.task, tree_str,
                dreamer=dreamer,
                current_url=session.browser.current_url(),
                num_plans=config.num_plans,
                min_steps=config.min_steps,
                scoring_model=config.scoring_model,
                session_context=session_context,
                event_callback=cb,
            )

            if dreamer is not None:
                del dreamer

            if not plan:
                _emit(session, "error", {"message": "No valid plans generated."})
                return

            if session.cancelled:
                raise InterruptedError("Cancelled")

            _emit(session, "status", {"phase": "executing", "message": f"Executing plan ({len(plan)} steps)..."})
            _emit(session, "execution_start", {"total_steps": len(plan)})

            for step_num, action in enumerate(plan):
                if session.cancelled:
                    raise InterruptedError("Cancelled")
                keep_going = _execute_step(session, step_num, action, len(plan), config, task=config.task)
                if not keep_going:
                    break

        _emit(session, "status", {"phase": "done", "message": "Task complete."})
        _emit(session, "done", {"message": "Agent finished."})

    except (InterruptedError, CancelledError):
        _emit(session, "done", {"message": "Cancelled by user."})
    except Exception as e:
        _emit(session, "error", {"message": str(e)})
    finally:
        if session.browser:
            try:
                session.browser.close()
            except Exception:
                pass
            session.browser = None
        with _session_lock:
            global _active_session
            _active_session = None


# --- API Endpoints ---

@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/run")
async def start_run(req: RunRequest):
    global _active_session
    with _session_lock:
        if _active_session is not None:
            raise HTTPException(409, "An agent session is already running.")
        session_id = uuid.uuid4().hex[:12]
        session = DashboardSession(session_id=session_id, config=req)
        _active_session = session

    thread = threading.Thread(target=run_dashboard_agent, args=(session,), daemon=True)
    session.thread = thread
    thread.start()

    return {"session_id": session_id}


@app.get("/api/events/{session_id}")
async def event_stream(session_id: str):
    with _session_lock:
        session = _active_session
    if session is None or session.session_id != session_id:
        raise HTTPException(404, "Session not found.")

    async def generate():
        while True:
            try:
                event = session.event_queue.get_nowait()
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in ("done", "error"):
                    break
            except queue.Empty:
                await asyncio.sleep(0.05)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/stop/{session_id}")
async def stop_session(session_id: str):
    with _session_lock:
        session = _active_session
    if session is None or session.session_id != session_id:
        raise HTTPException(404, "Session not found.")
    session.cancelled = True
    # Kill any in-flight LLM subprocesses immediately
    kill_action_llm()
    kill_scoring_llm()
    # Close browser to unblock any Playwright waits
    if session.browser:
        try:
            session.browser.close()
        except Exception:
            pass
        session.browser = None
    return {"status": "cancelling"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
