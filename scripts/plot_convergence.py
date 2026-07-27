"""
scripts/plot_convergence.py
===========================
Generates three charts from benchmark_results.json:

  1. Accuracy per round (all 4 ablation conditions)
  2. DP privacy budget ε per round (full-stack condition)
  3. Communication overhead comparison (bar chart)

Usage
-----
    python scripts/plot_convergence.py

Output files (in data/):
    convergence_accuracy.png
    dp_epsilon_per_round.png
    communication_overhead.png
"""

from __future__ import annotations

import json
import pathlib
import sys

DATA_DIR  = pathlib.Path(__file__).parent.parent / "data"
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = DATA_DIR / "benchmark_results.json"


def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        print(
            f"[ERROR] {RESULTS_FILE} not found.\n"
            "Run:  python app/benchmark.py\nto generate it first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(RESULTS_FILE) as f:
        return json.load(f)


def plot_accuracy(results: dict) -> None:
    """Plot accuracy per round for each ablation condition."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))

    conditions = results.get("ablation_conditions", {})
    for label, cond_data in conditions.items():
        rounds_data = cond_data.get("rounds", [])
        if not rounds_data:
            continue
        xs = [r["round"] for r in rounds_data]
        ys = [r.get("avg_accuracy", r.get("accuracy", 0.0)) for r in rounds_data]
        ax.plot(xs, ys, marker="o", label=label)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Accuracy")
    ax.set_title("Federated Learning Convergence — Accuracy per Round")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "convergence_accuracy.png"
    fig.savefig(out, dpi=150)
    print(f"✅  Saved: {out}")
    plt.close(fig)


def plot_epsilon(results: dict) -> None:
    """Plot cumulative ε per round for the full-stack condition."""
    import matplotlib.pyplot as plt

    full_stack = (
        results.get("ablation_conditions", {}).get("full_stack")
        or results.get("ablation_conditions", {}).get("FullStack+DP+SMPC")
        or next(iter(results.get("ablation_conditions", {}).values()), None)
    )
    if not full_stack:
        print("⚠️  No 'full_stack' condition found in benchmark results — skipping ε chart.")
        return

    rounds_data = full_stack.get("rounds", [])
    xs = [r["round"] for r in rounds_data]
    ys = [r.get("cumulative_epsilon", r.get("epsilon", 0.0)) for r in rounds_data]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, marker="s", color="darkorange", linewidth=2)
    ax.fill_between(xs, ys, alpha=0.15, color="darkorange")
    ax.set_xlabel("FL Round")
    ax.set_ylabel("Cumulative ε (privacy budget consumed)")
    ax.set_title("Differential Privacy Budget (ε) Consumed per Round")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "dp_epsilon_per_round.png"
    fig.savefig(out, dpi=150)
    print(f"✅  Saved: {out}")
    plt.close(fig)


def plot_comm_overhead(results: dict) -> None:
    """Bar chart: communication overhead in KB per round per condition."""
    import matplotlib.pyplot as plt

    overhead = results.get("communication_overhead_bytes", {})
    if not overhead:
        print("⚠️  No communication_overhead_bytes found — skipping overhead chart.")
        return

    labels = list(overhead.keys())
    values = [v / 1024 for v in overhead.values()]   # convert to KB

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#3b82d4", "#7c5cd8", "#e05c3a", "#2ea96a"])
    ax.bar_label(bars, fmt="%.1f KB", padding=3, fontsize=9)
    ax.set_ylabel("Bytes per round (KB)")
    ax.set_title("Communication Overhead per FL Round — Ablation Comparison")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = PLOTS_DIR / "communication_overhead.png"
    fig.savefig(out, dpi=150)
    print(f"✅  Saved: {out}")
    plt.close(fig)


def main() -> None:
    results = _load_results()
    plot_accuracy(results)
    plot_epsilon(results)
    plot_comm_overhead(results)
    print("\n📊  All charts saved to data/plots/")


if __name__ == "__main__":
    main()
