"""
Shared text-only LLM call utility.
Routes through claud CLI (proxy) or Anthropic API depending on environment.
"""

import os
import json
import subprocess

from action_generator import _has_claud_proxy, _get_claud_env, _proc_lock
import threading


# Track active subprocess for llm_call too
_active_proc: subprocess.Popen | None = None
_llm_proc_lock = threading.Lock()


def kill_active_llm_call():
    """Kill any in-flight claud subprocess from llm_call."""
    with _llm_proc_lock:
        if _active_proc is not None and _active_proc.poll() is None:
            _active_proc.kill()


def call_text_llm(system: str, user_prompt: str, max_tokens: int = 256) -> str:
    """Make a text-only LLM call (no images).

    Checks for claud CLI first, falls back to direct Anthropic API.
    """
    if _has_claud_proxy():
        return _call_via_claud(system, user_prompt)
    else:
        return _call_via_api(system, user_prompt, max_tokens)


def _call_via_claud(system: str, user_prompt: str) -> str:
    global _active_proc
    env = _get_claud_env()
    full_prompt = f"{system}\n\n{user_prompt}"
    with _llm_proc_lock:
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
        with _llm_proc_lock:
            _active_proc = None
    return stdout.strip()


def _call_via_api(system: str, user_prompt: str, max_tokens: int) -> str:
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        creds_path = os.path.expanduser("~/.claude/.credentials.json")
        if os.path.exists(creds_path):
            with open(creds_path) as f:
                creds = json.load(f)
            key = creds.get("claudeAiOauth", {}).get("accessToken")
    if not key:
        raise RuntimeError("No API key found")

    if key.startswith("sk-ant-oat"):
        client = anthropic.Anthropic(
            auth_token=key,
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
        )
    else:
        client = anthropic.Anthropic(api_key=key)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text.strip()
