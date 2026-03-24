#!/usr/bin/env python3
"""Generate standalone evaluation graphs from eval_results.json."""

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "eval_results.json"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib.pyplot as plt

with RESULTS_PATH.open() as f:
    data = json.load(f)

results = data["results"]
summary = data["summary"]

# Extract per-task paired data in the original evaluation order.
task_ids = []
seen = set()
for result in results:
    task_id = result["task_id"]
    if task_id not in seen:
        seen.add(task_id)
        task_ids.append(task_id)

lookup = {(result["task_id"], result["mode"]): result for result in results}

modes = ["reactive-dreamer", "plan-first"]
colors = {"reactive-dreamer": "#e74c3c", "plan-first": "#3498db"}
mode_labels = {
    "reactive-dreamer": "Reactive-Dreamer",
    "plan-first": "Plan-First",
}
task_tick_labels = [task_id.replace("_", " ").title() for task_id in task_ids]

plt.style.use("seaborn-v0_8-whitegrid")


def base_axes(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    return fig, ax


def save_plot(fig, filename):
    output_path = ROOT / filename
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path.name}")


def plot_overall_success_rate():
    fig, ax = base_axes((8, 6))
    x = np.arange(len(modes))
    success_rates = [summary[mode]["success_rate"] * 100 for mode in modes]
    bars = ax.bar(
        x,
        success_rates,
        color=[colors[mode] for mode in modes],
        edgecolor="white",
        width=0.55,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[mode] for mode in modes])
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Overall Pass Rate", fontweight="bold")
    for bar, value in zip(bars, success_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    save_plot(fig, "eval_pass_rate.png")


def plot_metric_per_task(metric_key, ylabel, title, filename):
    fig, ax = base_axes((12, 6))
    x = np.arange(len(task_ids))
    width = 0.35
    for idx, mode in enumerate(modes):
        values = [lookup[(task_id, mode)][metric_key] for task_id in task_ids]
        ax.bar(
            x + idx * width,
            values,
            width,
            color=colors[mode],
            label=mode_labels[mode],
            edgecolor="white",
        )
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(task_tick_labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(frameon=False)
    save_plot(fig, filename)


def plot_metric_average(summary_key, ylabel, title, filename, formatter):
    fig, ax = base_axes((8, 6))
    x = np.arange(len(modes))
    values = [summary[mode][summary_key] for mode in modes]
    bars = ax.bar(
        x,
        values,
        color=[colors[mode] for mode in modes],
        edgecolor="white",
        width=0.55,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([mode_labels[mode] for mode in modes])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    upper = max(values) * 1.2 if values else 1
    ax.set_ylim(0, upper)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + upper * 0.02,
            formatter(value),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    save_plot(fig, filename)


plot_overall_success_rate()
plot_metric_per_task(
    "steps_taken",
    ylabel="Steps",
    title="Steps Taken per Task",
    filename="eval_steps_taken.png",
)
plot_metric_average(
    "avg_steps",
    ylabel="Average Steps",
    title="Average Steps Taken",
    filename="eval_avg_steps_taken.png",
    formatter=lambda value: f"{value:.2f}",
)
plot_metric_per_task(
    "total_time_s",
    ylabel="Time (seconds)",
    title="Time Taken per Task",
    filename="eval_time_taken.png",
)
plot_metric_average(
    "avg_time_s",
    ylabel="Average Time (seconds)",
    title="Average Time Taken",
    filename="eval_avg_time_taken.png",
    formatter=lambda value: f"{value:.1f}s",
)
