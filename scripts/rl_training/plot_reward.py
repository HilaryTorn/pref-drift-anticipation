#!/usr/bin/env python3
"""Render the reward-vs-step curve for an RL run as a PNG.

Used two ways:
- live: DriftCadenceCallback calls ``plot_reward_curve`` during training, so
  the figure under results/<run_name>/ refreshes every few steps;
- post-hoc: ``python -m rl_training.plot_reward --run_dir runs/<run>`` rebuilds
  the same figure from trainer_state.json + reward_log.jsonl.

Two series share one bounded [0, 1] pass-rate axis: the online training reward
(mean over rollout groups, every logging_steps) and the held-out eval reward
(DriftCadenceCallback, every checkpoint).
"""

from __future__ import annotations

import json
from pathlib import Path

# Categorical slots 1-2 of the validated palette; text/grid stay neutral ink.
_TRAIN_COLOR = "#2a78d6"
_EVAL_COLOR = "#1baf7a"
_INK = "#3d3d3a"
_MUTED = "#83827d"
_GRID = "#e8e7e2"


def train_points_from_log_history(log_history: list[dict]) -> list[tuple[int, float]]:
    """(step, reward) from Trainer log entries that carry an online reward."""
    points = {}
    for entry in log_history:
        if "reward" in entry and "step" in entry:
            points[int(entry["step"])] = float(entry["reward"])
    return sorted(points.items())


def eval_points_from_reward_log(reward_log_path: str | Path) -> list[tuple[int, float]]:
    """(step, eval_reward) from reward_log.jsonl; reruns append, last one wins."""
    path = Path(reward_log_path)
    if not path.exists():
        return []
    points = {}
    with path.open() as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                points[int(row["step"])] = float(row["eval_reward"])
    return sorted(points.items())


def plot_reward_curve(
    train_points: list[tuple[int, float]],
    eval_points: list[tuple[int, float]],
    out_path: str | Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    if train_points:
        steps, rewards = zip(*train_points)
        # a single point draws no line segment; give it a visible marker
        marker = "o" if len(train_points) == 1 else None
        ax.plot(
            steps, rewards, color=_TRAIN_COLOR, linewidth=1.8,
            marker=marker, markersize=6, label="train reward (rollout mean)",
        )
    if eval_points:
        steps, rewards = zip(*eval_points)
        ax.plot(
            steps, rewards, color=_EVAL_COLOR, linewidth=1.8,
            marker="o", markersize=6, label="eval reward (held-out)",
        )
        ax.annotate(
            f"{rewards[-1]:.3f}", (steps[-1], rewards[-1]),
            textcoords="offset points", xytext=(6, 6), fontsize=9, color=_INK,
        )

    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.0)  # reward is a pass-rate fraction; keep the honest bounded scale
    ax.set_xlabel("optimizer step", color=_INK)
    ax.set_ylabel("verifier pass rate", color=_INK)
    ax.set_title(title, color=_INK, fontsize=11, loc="left")
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_GRID)
    ax.tick_params(colors=_MUTED, labelsize=9)
    if train_points or eval_points:
        ax.legend(frameon=False, fontsize=9, labelcolor=_INK, loc="upper left")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def latest_trainer_state(run_dir: Path) -> list[dict]:
    checkpoints = sorted(
        run_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]),
    )
    if not checkpoints:
        return []
    return json.loads(checkpoints[-1].read_text()).get("log_history", [])


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="runs/<run_name> directory")
    parser.add_argument("--out", default=None, help="output PNG (default: results/<run_name>/reward_curve.png)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out = Path(args.out) if args.out else Path("results") / run_dir.name / "reward_curve.png"
    train_points = train_points_from_log_history(latest_trainer_state(run_dir))
    eval_points = eval_points_from_reward_log(run_dir / "reward_log.jsonl")
    plot_reward_curve(train_points, eval_points, out, title=f"{run_dir.name} — verifiable reward")
    print(f"[plot] {len(train_points)} train points, {len(eval_points)} eval points -> {out}")


if __name__ == "__main__":
    main()
