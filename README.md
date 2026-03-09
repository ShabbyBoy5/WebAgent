# WebAgent

A web browsing agent that uses **Claude** for action generation and **Dreamer-7B** as a world model for look-ahead planning, all driven by **Playwright** in a real browser.

Two modes:
- **Reactive** — Claude sees the page and picks the best next action (fast, ~2s/step)
- **Planning** — Claude proposes multiple candidates, Dreamer-7B predicts outcomes, Claude scores them (~20s/step, smarter)

## Setup

Requires Python 3.12 and a GPU with ~5GB VRAM (for Dreamer-7B planning mode).

```bash
# Create venv
uv venv --python 3.12 ~/webagent-env
source ~/webagent-env/bin/activate

# Install dependencies
pip install anthropic playwright pillow torch transformers bitsandbytes accelerate qwen-vl-utils

# Install browser
python -m playwright install chromium
```

### Authentication

The agent reads your Anthropic API key in this order:
1. `ANTHROPIC_API_KEY` environment variable
2. Claude Code OAuth credentials at `~/.claude/.credentials.json`

If using an API key directly:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Usage

### Reactive mode (Claude only, no GPU needed)

```bash
python agent.py \
  --task "Search for red blankets" \
  --url "https://www.amazon.com" \
  --max-steps 10
```

### Planning mode (Claude + Dreamer-7B)

```bash
python agent.py \
  --task "Find the cheapest noise-cancelling wireless headphones with at least 4 stars, add it to the cart" \
  --url "https://www.amazon.com" \
  --max-steps 15 \
  --planning
```

### Headless (no visible browser window)

```bash
python agent.py \
  --task "Search for red blankets" \
  --url "https://www.amazon.com" \
  --max-steps 10 \
  --headless
```

### All flags

| Flag | Description | Default |
|------|-------------|---------|
| `--task` | Task to accomplish (required) | — |
| `--url` | Starting URL (required) | — |
| `--max-steps` | Maximum agent steps | 15 |
| `--headless` | Run browser without a window | off |
| `--planning` | Enable Dreamer-7B planning pipeline | off |

## How it works

### Reactive mode
```
screenshot + accessibility tree → Claude picks best action → execute in browser → repeat
```

### Planning mode
```
screenshot + accessibility tree
  → Claude proposes 3-5 candidate actions
  → Dreamer-7B predicts page state after each action
  → Claude scores each (action, predicted state) pair
  → execute the highest-scored action
  → repeat
```

## Architecture

| File | Purpose |
|------|---------|
| `agent.py` | Main agent loop and CLI |
| `action_generator.py` | Claude-based action generation (single or multi-candidate) |
| `planning.py` | Planning pipeline: candidates → Dreamer prediction → Claude scoring |
| `browser_executor.py` | Playwright browser management, accessibility tree extraction |
| `dreamer_model.py` | Dreamer-7B world model with 4-bit quantization |

### Original WebDreamer modules (from the paper)

| File | Purpose |
|------|---------|
| `world_model.py` | WebWorldModel using GPT-4o/Claude |
| `simulation_scoring.py` | Simulation-based action scoring |
| `controller.py` | Action filtering and selection |
| `llms/` | LLM provider utilities |

## Based on

[WebDreamer: Model-Based Planning for Web Agents](https://arxiv.org/abs/2411.06559) by OSU NLP Group.

This repo extends the original with a working Playwright-based agent that can navigate real websites using Claude for decisions and Dreamer-7B for planning.
