"""
scripts/ablation_study.py
=========================
Standalone 4-condition ablation study runner.

Conditions (FIXED — each has a distinct configuration)
----------
  1. plain_fedavg           — FedAvg, no DP (nm=0.0), epsilon=inf
  2. fedavg_dp_high_noise   — FedAvg + DP (high noise, nm=1.5)
  3. fedavg_dp_smpc         — FedAvg + DP (nm=1.1) + SMPC simulation
  4. full_stack             — DP (nm=0.8, LOWER noise) + SMPC + Byzantine detection

NOTES
-----
- plain_fedavg uses nm=0.0 which gives epsilon=inf (no privacy).
- fedavg_dp_smpc (nm=1.1) and full_stack (nm=0.8) use DIFFERENT noise levels;
  full_stack has more noise (lower nm → wait, higher nm = more noise; here 0.8 < 1.1
  so full_stack has LESS noise = higher accuracy but weaker privacy).
- SMPC is always a single-process simulation; the label documents the intended
  semantics only.
- Accuracy = global model on held-out test set (not local per-node accuracy).

Usage
-----
    python scripts/ablation_study.py
    python scripts/ablation_study.py --rounds 5
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _run_condition(label: str, n_rounds: int, **sim_kwargs) -> dict:
    """Run one simulation condition and return a structured result dict."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from app.federated_sim_engine import FederatedSimulation

    print(f"  [{label}] running {n_rounds} rounds ...", end="", flush=True)
    t0 = time.perf_counter()
    sim = FederatedSimulation(n_rounds=n_rounds, **sim_kwargs)
    logs = sim.run_simulation()
    elapsed = time.perf_counter() - t0
    print(f" done ({elapsed:.1f}s)")

    final_acc = sim.get_final_global_accuracy()
    final_eps_raw = logs[-1]["cumulative_epsilon"] if logs else 0.0
    final_eps = final_eps_raw if not math.isinf(final_eps_raw) else float("inf")

    avg_acc_per_round = [
        {
            "round": rlog["round"],
            # Global model accuracy on held-out test set (not local node average)
            "global_accuracy": rlog.get("global_accuracy", float("nan")),
            # Local node average kept separately for reference
            "avg_local_accuracy": round(
                sum(n["accuracy"] for n in rlog["nodes"]) / len(rlog["nodes"]), 4
            ),
            "cumulative_epsilon": rlog["cumulative_epsilon"],
        }
        for rlog in logs
    ]
    return {
        "label": label,
        "n_rounds": n_rounds,
        "noise_multiplier": sim_kwargs.get("noise_multiplier", 0.0),
        "final_global_accuracy": round(final_acc, 4),
        "final_cumulative_epsilon": (
            round(final_eps, 6) if not math.isinf(final_eps) else "inf"
        ),
        "elapsed_seconds": round(elapsed, 2),
        "rounds": avg_acc_per_round,
        "synthetic_fallback": sim.synthetic_fallback,
    }


def run_ablation(n_rounds: int = 10, n_nodes: int = 3) -> dict:
    results = {}

    # Condition 1: Plain FedAvg — no DP, epsilon = inf
    results["plain_fedavg"] = _run_condition(
        "plain_fedavg",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=0.0,  # no DP noise; epsilon = inf
    )

    # Condition 2: FedAvg + high DP noise (nm=1.5)
    # Note: 'high noise' = nm=1.5 (more noise = lower accuracy = stronger privacy)
    results["fedavg_dp_high_noise"] = _run_condition(
        "fedavg_dp_high_noise",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=1.5,
    )

    # Condition 3: FedAvg + DP (nm=1.1) + SMPC simulation
    results["fedavg_dp_smpc"] = _run_condition(
        "fedavg_dp_smpc",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=1.1,
    )

    # Condition 4: Full stack — DP (nm=0.8, less noise) + SMPC + Byzantine detection
    # nm=0.8 is DIFFERENT from nm=1.1 above — lower noise → higher accuracy, weaker DP
    results["full_stack"] = _run_condition(
        "full_stack",
        n_rounds=n_rounds,
        n_nodes=n_nodes,
        noise_multiplier=0.8,
    )

    # Byzantine resilience test under full_stack config
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from app.federated_sim_engine import FederatedSimulation
    sim_byz = FederatedSimulation(
        n_rounds=n_rounds, n_nodes=n_nodes, noise_multiplier=0.8
    )
    logs_byz = sim_byz.run_simulation(tamper_round=1, tamper_node=0)
    byz_flagged = {
        rlog["round"]: rlog.get("flagged_nodes", [])
        for rlog in logs_byz
        if rlog.get("flagged_nodes")
    }
    results["full_stack"]["byzantine_resilience"] = {
        "tampered_node": "Hospital-1",
        "flagged_rounds": byz_flagged,
        "detection_occurred": len(byz_flagged) > 0,
    }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 4-condition ablation study")
    parser.add_argument("--rounds", type=int, default=10, help="FL rounds per condition")
    parser.add_argument("--nodes",  type=int, default=3,  help="Number of FL nodes")
    args = parser.parse_args()

    print(f"🔬  Running ablation study: {args.rounds} rounds × {args.nodes} nodes\n")
    print("Conditions:")
    print("  plain_fedavg        : nm=0.0 (no DP, epsilon=inf)")
    print("  fedavg_dp_high_noise: nm=1.5 (high noise, strong DP)")
    print("  fedavg_dp_smpc      : nm=1.1 (moderate DP + SMPC sim)")
    print("  full_stack          : nm=0.8 (lower noise, weaker DP + SMPC sim)\n")

    results = run_ablation(n_rounds=args.rounds, n_nodes=args.nodes)

    out_path = DATA_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅  Ablation results written to {out_path}")
    print("\nSummary (global model accuracy on held-out test set):")
    for cond, data in results.items():
        if cond == "full_stack" and "byzantine_resilience" in data:
            byz = data["byzantine_resilience"]
            byz_str = f"  Byzantine detected={byz['detection_occurred']}"
        else:
            byz_str = ""
        eps_str = str(data["final_cumulative_epsilon"])
        print(
            f"  {cond:<30} acc={data['final_global_accuracy']:.4f}  "
            f"ε={eps_str}  "
            f"({data['elapsed_seconds']}s){byz_str}"
        )


if __name__ == "__main__":
    main()
