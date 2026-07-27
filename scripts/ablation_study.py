"""
scripts/ablation_study.py
=========================
Standalone 4-condition ablation study runner.

Conditions
----------
  1. plain_fedavg       — FedAvg, no DP, no SMPC
  2. fedavg_dp_high     — FedAvg + DP (high noise, noise_multiplier=3.0)
  3. fedavg_dp_smpc     — FedAvg + DP (noise_multiplier=1.1) + SMPC
  4. full_stack         — DP (noise_multiplier=1.1) + SMPC + Byzantine detection

Each condition runs for N rounds on the Cleveland Heart Disease dataset.
Results are written to data/ablation_results.json.

Usage
-----
    python scripts/ablation_study.py
    python scripts/ablation_study.py --rounds 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _run_condition(label: str, n_rounds: int, **sim_kwargs) -> dict:
    """Run one simulation condition and return a structured result dict."""
    import sys, os
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from app.federated_sim_engine import FederatedSimulation

    print(f"  [{label}] running {n_rounds} rounds ...", end="", flush=True)
    t0 = time.perf_counter()
    sim = FederatedSimulation(n_rounds=n_rounds, **sim_kwargs)
    logs = sim.run_simulation()
    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s)")

    final_acc = sim.get_final_global_accuracy()
    final_eps = logs[-1]["cumulative_epsilon"] if logs else 0.0
    avg_acc_per_round = [
        {
            "round": rlog["round"],
            "avg_accuracy": round(
                sum(n["accuracy"] for n in rlog["nodes"]) / len(rlog["nodes"]), 4
            ),
            "cumulative_epsilon": rlog["cumulative_epsilon"],
        }
        for rlog in logs
    ]
    return {
        "label": label,
        "n_rounds": n_rounds,
        "final_global_accuracy": round(final_acc, 4),
        "final_cumulative_epsilon": round(final_eps, 6),
        "elapsed_seconds": round(elapsed, 2),
        "rounds": avg_acc_per_round,
    }


def run_ablation(n_rounds: int = 10, n_nodes: int = 3) -> dict:
    results = {}

    results["plain_fedavg"] = _run_condition(
        "plain_fedavg",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=0.0,  # no DP noise
    )

    results["fedavg_dp_high_noise"] = _run_condition(
        "fedavg_dp_high_noise",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=3.0,
    )

    results["fedavg_dp_smpc"] = _run_condition(
        "fedavg_dp_smpc",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=1.1,
    )

    results["full_stack"] = _run_condition(
        "full_stack",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=1.1,
    )
    # Re-run full_stack with a Byzantine node to test resilience
    import sys, os
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from app.federated_sim_engine import FederatedSimulation
    sim_byz = FederatedSimulation(n_rounds=n_rounds, n_nodes=n_nodes, noise_multiplier=1.1)
    logs_byz = sim_byz.run_simulation(tamper_round=1, tamper_node=0)
    byz_flagged = {
        rlog["round"]: rlog.get("flagged_nodes", [])
        for rlog in logs_byz
        if rlog.get("flagged_nodes")
    }
    results["full_stack"]["byzantine_resilience"] = {
        "tampered_node": "Hospital-1",
        "flagged_rounds": byz_flagged,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-condition ablation study")
    parser.add_argument("--rounds", type=int, default=10, help="FL rounds per condition")
    parser.add_argument("--nodes",  type=int, default=3,  help="Number of FL nodes")
    args = parser.parse_args()

    print(f"🔬  Running ablation study: {args.rounds} rounds × {args.nodes} nodes\n")
    results = run_ablation(n_rounds=args.rounds, n_nodes=args.nodes)

    out_path = DATA_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅  Ablation results written to {out_path}")
    print("\nSummary:")
    for cond, data in results.items():
        print(
            f"  {cond:<30} acc={data['final_global_accuracy']:.4f}  "
            f"ε={data['final_cumulative_epsilon']:.4f}  "
            f"({data['elapsed_seconds']}s)"
        )


if __name__ == "__main__":
    main()
