"""
benchmark.py
============
End-to-end benchmarking suite for the Blockchain-FL framework.

Runs four experimental conditions and measures:
  - Model accuracy (accuracy + AUC-ROC) per round on HELD-OUT TEST SET
  - Real differential privacy budget (ε) consumed per round
  - Communication overhead (bytes exchanged per round)
  - Poisoning attack resilience (accuracy degradation under Byzantine node)

All results are computed from real FL execution — nothing is hard-coded.

Usage
-----
    python -m app.benchmark                  # full benchmark
    python -m app.benchmark --rounds 5       # fast run

Output
------
    data/benchmark_results.json    (machine-readable)
    data/benchmark_summary.txt     (human-readable)

NOTES
-----
- Accuracy values reported are the GLOBAL model on the held-out test set
  (not local per-node accuracy on each node's own split).
- DP epsilon values are computed with replace-one sensitivity (factor=2),
  giving honest (larger) epsilon values.
- Epsilon for noise_multiplier=0 is mathematically infinite (no privacy);
  shown as 'inf' in the JSON.
- 'non-IID' sharding is actually near-IID (StratifiedKFold); see engine docs.
- SMPC is a single-process simulation; no cryptographic isolation.
- numpy.Generator is NOT a CSPRNG; these are research results only.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import sys
import time
from typing import Any

import numpy as np

from app.federated_sim_engine import FederatedSimulation


# ── Output directory ──────────────────────────────────────────────────────────
DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


# ── Centralised baseline ──────────────────────────────────────────────────────

def _centralized_baseline(seed: int = 42) -> dict[str, float]:
    """
    Train a single centralised model on all training data (upper-bound reference).
    Uses the same train/test split strategy as FederatedSimulation (20% held out).
    Preprocessing fitted on training data only — no leakage.
    Returns accuracy and AUC-ROC on held-out test set.
    """
    from app.federated_sim_engine import _load_cleveland, _preprocess
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X_raw, y = _load_cleveland()
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_te = _preprocess(X_tr_raw, X_te_raw)
    model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "auc_roc": round(float(roc_auc_score(y_te, probs)), 4),
    }


def _local_only_baseline(seed: int = 42, n_nodes: int = 3) -> dict[str, float]:
    """
    Local-only (no federation) baseline: each node trains independently on 80 %
    of its own shard (the same local split the federated nodes use), and we
    report the mean test accuracy of those local models on the global held-out
    set. ``accuracy_std`` here is the spread ACROSS NODES for this one seed.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from app.federated_sim_engine import _load_cleveland, _preprocess, _heterogeneous_split

    X_raw, y = _load_cleveland()
    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=seed, stratify=y
    )
    X_tr, X_te = _preprocess(X_tr_raw, X_te_raw)
    splits = _heterogeneous_split(X_tr, y_tr, n_nodes, seed=seed)
    accs = []
    for X_shard, y_shard in splits:
        X_s_tr, _, y_s_tr, _ = train_test_split(
            X_shard, y_shard, test_size=0.2, random_state=seed, stratify=y_shard
        )
        model = LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed)
        model.fit(X_s_tr, y_s_tr)
        preds = model.predict(X_te)
        accs.append(float(accuracy_score(y_te, preds)))
    return {
        "accuracy": round(float(np.mean(accs)), 4),
        "accuracy_std": round(float(np.std(accs)), 4),
    }


def _across_seeds(per_seed: list[dict[str, float]], seeds: list[int],
                  keys: tuple[str, ...]) -> dict[str, Any]:
    """Mean and standard deviation over seeds (same definition as the federated rows)."""
    out: dict[str, Any] = {"n_seeds": len(per_seed), "seeds": list(seeds), "per_seed": per_seed}
    for k in keys:
        vals = [r[k] for r in per_seed]
        out[k] = round(float(np.mean(vals)), 4)
        out[f"{k}_std"] = round(float(np.std(vals)), 4)
    return out


# ── Communication overhead ────────────────────────────────────────────────────

def _comm_overhead_bytes(n_nodes: int, n_features: int) -> int:
    """
    ANALYTICAL ESTIMATE, not a measurement (nothing is sent over a network):
      - Each node uploads (n_features + 1) float64 parameters
      - Total = n_nodes * (n_features + 1) * 8
      - The factor 2 models one share exchange and is applied to every
        condition, including those that do not use secret sharing.
    """
    params_per_node = (n_features + 1) * 8  # float64
    return n_nodes * params_per_node * 2     # *2 for SMPC share exchange


# ── Four ablation conditions ──────────────────────────────────────────────────

def _run_condition(
    label: str,
    n_nodes: int,
    n_rounds: int,
    noise_multiplier: float,
    use_smpc: bool,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Run one ablation condition and return per-round metrics.
    use_smpc=True forms the clipped-delta sum via additive secret sharing
    (single-process arithmetic simulation; the result equals the plain sum).
    noise_multiplier=0 means no DP (epsilon=inf).

    Per-round accuracy = GLOBAL model on held-out test set (not local node acc).
    """
    print(f"  [{label}] running {n_rounds} rounds …", flush=True)
    t0 = time.time()

    # noise_multiplier=0 → pass 0.0 directly; engine handles inf epsilon correctly
    nm = noise_multiplier
    sim = FederatedSimulation(
        n_nodes=n_nodes,
        n_rounds=n_rounds,
        noise_multiplier=nm,
        max_grad_norm=1.0,
        delta=1e-5,
        seed=seed,
        use_smpc=use_smpc,
    )
    logs = sim.run_simulation()

    per_round_acc: list[float] = []
    per_round_auc: list[float] = []
    per_round_eps: list[Any] = []
    per_round_comm: list[int] = []
    per_round_local_acc: list[float] = []  # local node average (kept separately)

    for log in logs:
        # Global model accuracy on held-out test set
        global_acc = log.get("global_accuracy", float("nan"))
        global_auc = log.get("global_auc_roc", float("nan"))
        per_round_acc.append(round(float(global_acc), 4))
        per_round_auc.append(round(float(global_auc) if not math.isnan(global_auc) else 0.0, 4))

        # Local node average (kept as informational field, not headline)
        node_accs = [n["accuracy"] for n in log["nodes"]]
        per_round_local_acc.append(round(float(np.mean(node_accs)), 4))

        # ε: use actual epsilon; inf for no-DP case
        eps_val = log["cumulative_epsilon"] if noise_multiplier > 0 else float("inf")
        per_round_eps.append(
            round(eps_val, 6) if not math.isinf(eps_val) else "inf"
        )
        per_round_comm.append(_comm_overhead_bytes(n_nodes, sim.n_features))

    elapsed = round(time.time() - t0, 2)
    final_global_acc = sim.get_final_global_accuracy()
    final_eps = per_round_eps[-1] if per_round_eps else ("inf" if noise_multiplier == 0 else 0.0)

    return {
        "label": label,
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
        "noise_multiplier": noise_multiplier,
        "use_smpc": use_smpc,
        "seed": seed,
        "dataset": "UCI Cleveland Heart Disease (n=303, 13 features)",
        # Headline accuracy: global model on held-out test set
        "per_round_accuracy": per_round_acc,
        "per_round_auc_roc": per_round_auc,
        # Informational: local node average (NOT the headline metric)
        "per_round_local_node_accuracy": per_round_local_acc,
        "per_round_epsilon": per_round_eps,
        "per_round_comm_bytes": per_round_comm,
        "final_global_accuracy": round(final_global_acc, 4),
        "final_epsilon": final_eps,
        "global_hashes": sim.global_hashes,
        "elapsed_s": elapsed,
        "synthetic_fallback": sim.synthetic_fallback,
    }


# ── Byzantine resilience test ─────────────────────────────────────────────────

def _run_byzantine_test(n_nodes: int, n_rounds: int, seed: int = 42) -> dict[str, Any]:
    """
    Inject one Byzantine node at round 3 and measure accuracy degradation + detection.
    Uses max(n_nodes, 4) nodes so the norm-based outlier detector has enough
    reference points (n=3 is underpowered for IQR-based detection with 1 outlier).
    """
    print("  [Byzantine Resilience] running …", flush=True)
    detect_nodes = max(n_nodes, 4)

    sim = FederatedSimulation(
        n_nodes=detect_nodes, n_rounds=n_rounds,
        noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5, seed=seed,
    )
    logs = sim.run_simulation(tamper_round=2, tamper_node=0)

    flagged = []
    accs = []
    for log in logs:
        flagged.extend(log.get("flagged_nodes", []))
        accs.append(round(float(log.get("global_accuracy", float("nan"))), 4))

    return {
        "per_round_global_accuracy": accs,
        "flagged_nodes": list(set(flagged)),
        "poisoning_detected": len(flagged) > 0,
    }


# ── Multi-seed aggregation ────────────────────────────────────────────────────

def _aggregate_seeds(runs: list[dict]) -> dict[str, Any]:
    """
    Given a list of per-seed condition results, compute mean ± std
    over final_global_accuracy and final_epsilon (numeric only).
    """
    accs = [r["final_global_accuracy"] for r in runs]
    # Handle inf epsilon for no-DP case
    epss_numeric = [
        r["final_epsilon"] for r in runs
        if isinstance(r["final_epsilon"], (int, float))
        and not math.isinf(r["final_epsilon"])
    ]
    n = len(runs)
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))
    # 95% CI via ±1.96 * std/sqrt(n)
    ci95_acc = 1.96 * std_acc / math.sqrt(n) if n > 1 else float("nan")

    eps_mean = float(np.mean(epss_numeric)) if epss_numeric else float("inf")
    eps_std  = float(np.std(epss_numeric))  if epss_numeric else 0.0

    return {
        "mean_accuracy":  round(mean_acc, 4),
        "std_accuracy":   round(std_acc,  4),
        "ci95_accuracy":  round(ci95_acc, 4) if not math.isnan(ci95_acc) else "nan",
        "mean_epsilon":   round(eps_mean, 6) if not math.isinf(eps_mean) else "inf",
        "std_epsilon":    round(eps_std,  6),
        "n_seeds":        n,
        "seeds_used":     [r.get("seed", "?") for r in runs],
    }


# ── Library versions ──────────────────────────────────────────────────────────

def _lib_versions() -> dict[str, str]:
    """Collect versions of key libraries for reproducibility metadata."""
    import importlib
    versions = {}
    for lib in ["numpy", "sklearn", "scipy", "pandas"]:
        try:
            m = importlib.import_module(lib)
            versions[lib] = getattr(m, "__version__", "unknown")
        except ImportError:
            versions[lib] = "not installed"
    versions["python"] = sys.version.split()[0]
    return versions


def _git_commit() -> str:
    """Return current git commit hash (read-only)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(pathlib.Path(__file__).parent.parent),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _git_dirty() -> bool | None:
    """True if tracked files differ from HEAD, None if git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True,
            cwd=str(pathlib.Path(__file__).parent.parent),
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else None
    except Exception:
        return None


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(
    n_nodes: int = 3,
    n_rounds: int = 10,
    seed: int = 42,
    n_seeds: int = 5,
) -> dict[str, Any]:
    """
    Full benchmark across four ablation conditions + Byzantine test.
    Runs each condition n_seeds times with different random seeds and
    reports mean ± std ± 95% CI for publication-ready statistics.

    Conditions (FIXED ablation — each distinct)
    ----------
    1. Plain FedAvg          (nm=0.0,  no DP;   epsilon=inf)
    2. FedAvg + DP high      (nm=1.5,  high DP; more noise, lower accuracy)
    3. FedAvg + DP + SMPC    (nm=1.1,  moderate DP)
    4. Full stack            (nm=0.8,  lower noise, higher accuracy, DP+SMPC)

    Note: conditions 3 and 4 use DIFFERENT noise multipliers (1.1 vs 0.8) and
    thus have different epsilon values.  This was a bug in the previous version.

    Accuracy = GLOBAL model on held-out test set (no leakage).
    """
    print("=" * 60)
    print("Blockchain-FL Benchmark — All Conditions")
    print(f"  n_nodes={n_nodes}  n_rounds={n_rounds}  n_seeds={n_seeds}")
    print("=" * 60)

    t_start = time.time()

    seeds = [seed + i * 7 for i in range(n_seeds)]

    # Baselines use the SAME seeds (hence the same train/test splits and shards)
    # as the federated conditions, so every comparison is paired and like-for-like.
    print("Computing centralized baseline …")
    centralized = _across_seeds([_centralized_baseline(seed=s) for s in seeds], seeds,
                                ("accuracy", "auc_roc"))
    print(f"  Centralized accuracy={centralized['accuracy']} ±{centralized['accuracy_std']}"
          f"  AUC={centralized['auc_roc']}")

    print("Computing local-only baseline …")
    local_only = _across_seeds([_local_only_baseline(seed=s, n_nodes=n_nodes) for s in seeds],
                               seeds, ("accuracy",))
    print(f"  Local-only mean accuracy={local_only['accuracy']}  ±{local_only['accuracy_std']}")

    # Ablation conditions: (label, noise_multiplier, use_smpc_label)
    conditions = [
        ("Plain-FedAvg",         0.0,  False),
        ("FedAvg+DP(nm=1.5)",    1.5,  False),
        ("FedAvg+DP+SMPC(nm=1.1)", 1.1, True),
        ("FullStack(nm=0.8)",    0.8,  True),
    ]

    results_by_condition: list[dict] = []
    for label, nm, smpc in conditions:
        print(f"\n  [{label}] running {n_seeds} seeds …")
        seed_runs: list[dict] = []
        for s in seeds:
            res = _run_condition(label, n_nodes, n_rounds, nm, smpc, seed=s)
            res["seed"] = s
            seed_runs.append(res)
            eps_display = (
                f"{res['final_epsilon']:.4f}"
                if isinstance(res["final_epsilon"], float) and not math.isinf(res["final_epsilon"])
                else str(res["final_epsilon"])
            )
            print(
                f"    seed={s}  global_acc={res['final_global_accuracy']:.4f}  "
                f"ε={eps_display}  ({res['elapsed_s']}s)"
            )
        stats = _aggregate_seeds(seed_runs)
        ci_str = (
            f"{stats['ci95_accuracy']:.4f}"
            if isinstance(stats["ci95_accuracy"], float) else str(stats["ci95_accuracy"])
        )
        print(
            f"  → mean_global_acc={stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}"
            f"  (95% CI ±{ci_str})"
            f"  ε_mean={stats['mean_epsilon']}"
        )
        results_by_condition.append({
            "label":      label,
            "statistics": stats,
            "seed_runs":  seed_runs,
        })

    # Byzantine resilience (single representative seed)
    print("\n  [Byzantine Resilience] running …")
    byzantine = _run_byzantine_test(n_nodes, n_rounds, seed=seed)
    print(f"  Byzantine detected={byzantine['poisoning_detected']}  "
          f"flagged={byzantine['flagged_nodes']}")

    total_elapsed = round(time.time() - t_start, 2)

    full_stack_stats = results_by_condition[3]["statistics"]
    accuracy_gap = round(centralized["accuracy"] - full_stack_stats["mean_accuracy"], 4)

    full_stack_repr = results_by_condition[3]["seed_runs"][-1]

    # Get Cleveland dataset row count from a single load
    try:
        from app.federated_sim_engine import _load_cleveland
        X_raw, _ = _load_cleveland()
        n_rows_cleveland = int(X_raw.shape[0])
    except Exception:
        n_rows_cleveland = "unknown"

    output = {
        "metadata": {
            "framework":       "Blockchain-powered Federated Learning",
            "dataset":         f"UCI Cleveland Heart Disease (n={n_rows_cleveland}, 13 features, binary)",
            "n_rows":          n_rows_cleveland,
            "n_nodes":         n_nodes,
            "n_rounds":        n_rounds,
            "n_seeds":         n_seeds,
            "seeds":           seeds,
            "total_elapsed_s": total_elapsed,
            "dp_accountant":   (
                "Rényi DP (Mironov 2017), replace-one adjacency, "
                "sensitivity=2C, sampling_rate=1 (no subsampling amplification). "
                "NOT per-sample DP-SGD. Noise source: numpy.Generator (NOT CSPRNG)."
            ),
            "smpc": (
                "Single-process arithmetic simulation of additive secret sharing. "
                "No cryptographic isolation. Does not protect raw gradients "
                "from an in-process observer."
            ),
            "accuracy_metric": (
                "Global model evaluated on a held-out test set (20%); "
                "preprocessing (StandardScaler, median fill) fitted only on "
                "training data (no leakage). Shard split: StratifiedKFold (near-IID)."
            ),
            "git_commit":  _git_commit(),
            "git_dirty":   _git_dirty(),
            "lib_versions": _lib_versions(),
            "command": (
                f"python -m app.benchmark --nodes {n_nodes} --rounds {n_rounds} "
                f"--seed {seed} --n-seeds {n_seeds}"
            ),
            "synthetic_fallback": (
                full_stack_repr.get("synthetic_fallback", False)
            ),
        },
        "centralized_baseline": centralized,
        "local_only_baseline":  local_only,
        "accuracy_gap_vs_centralized": accuracy_gap,
        "ablation_conditions": {
            c["label"]: c["statistics"]
            for c in results_by_condition
        },
        "ablation_full": results_by_condition,
        "byzantine_resilience": byzantine,
        # Convenience top-level fields (backward-compatible with plot_convergence.py)
        "accuracy":      full_stack_repr["per_round_accuracy"],
        "auc_roc":       full_stack_repr["per_round_auc_roc"],
        "privacy_loss":  full_stack_repr["per_round_epsilon"],
        "communication": full_stack_repr["per_round_comm_bytes"],
        "communication_overhead_bytes": {
            c["label"]: c["seed_runs"][-1]["per_round_comm_bytes"][-1]
            for c in results_by_condition
        },
        "final_global_hash": full_stack_repr["global_hashes"][-1]
                             if full_stack_repr["global_hashes"] else None,
        "final_accuracy": full_stack_stats["mean_accuracy"],
        "final_epsilon":  full_stack_stats["mean_epsilon"],
    }

    # Persist to data/
    out_path = DATA_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n✅ Benchmark complete — results → {out_path}")

    # Human-readable summary
    summary_lines = [
        "Blockchain-FL Benchmark Summary",
        "=" * 65,
        f"Dataset         : {output['metadata']['dataset']}",
        f"Nodes / Rounds  : {n_nodes} / {n_rounds}",
        f"Seeds           : {n_seeds} (mean ± std reported, 95% CI shown)",
        "DP accountant   : Rényi DP, replace-one sensitivity (honest)",
        f"Centralized acc : {centralized['accuracy']} ± {centralized['accuracy_std']} "
        f"(AUC {centralized['auc_roc']}), mean ± std over seeds",
        f"Local-only acc  : {local_only['accuracy']} ± {local_only['accuracy_std']}, "
        "mean ± std over seeds",
        "",
        f"{'Condition':<35} {'mean_acc':>9} {'±std':>7}  {'95%CI':>7}  {'mean_ε':>12}",
        "-" * 65,
    ]
    for c in results_by_condition:
        s = c["statistics"]
        ci_str = (
            f"±{s['ci95_accuracy']:.4f}"
            if isinstance(s["ci95_accuracy"], float) else "N/A"
        )
        eps_str = (
            f"{s['mean_epsilon']:.4f}"
            if isinstance(s["mean_epsilon"], float) and not math.isinf(float(s["mean_epsilon"]))
            else str(s["mean_epsilon"])
        )
        summary_lines.append(
            f"  {c['label']:<33} {s['mean_accuracy']:>9.4f} {s['std_accuracy']:>7.4f}"
            f"  {ci_str:>7}  {eps_str:>12}"
        )
    summary_lines += [
        "",
        f"Accuracy gap vs centralized  : {accuracy_gap} (full stack, mean over seeds)",
        f"Byzantine poisoning detected : {byzantine['poisoning_detected']}",
        f"Total benchmark time         : {total_elapsed}s",
        "",
        "LIMITATIONS",
        "-" * 65,
        "- DP is client-level (not per-sample DP-SGD); epsilon values are large.",
        "- SMPC is a single-process simulation; no cryptographic isolation.",
        "- Dataset n≤303: confidence intervals will be wide.",
        "- 'non-IID' sharding is StratifiedKFold (near-IID); see engine docs.",
        "- numpy.Generator is NOT a CSPRNG.",
    ]
    summary_text = "\n".join(summary_lines)
    (DATA_DIR / "benchmark_summary.txt").write_text(summary_text)
    print(summary_text)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes",   type=int, default=3)
    parser.add_argument("--rounds",  type=int, default=10)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of independent seeds (default 5)")
    args = parser.parse_args()
    run_benchmark(
        n_nodes=args.nodes, n_rounds=args.rounds,
        seed=args.seed, n_seeds=args.n_seeds,
    )
