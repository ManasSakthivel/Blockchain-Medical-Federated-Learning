# Results

All numbers come from `python -m app.benchmark --nodes 3 --rounds 10 --seed 42 --n-seeds 5`
(runtime under one second) on the 303-row Cleveland data. The raw output is
`data/benchmark_results.json` (with a human summary in `data/benchmark_summary.txt`);
its `metadata` block records the git commit, whether the tracked tree was dirty,
library versions, seeds and the exact command. **Only a result file whose
`git_dirty` is `false` is a clean reproduction of the code at `git_commit`.**
Regenerate with `make benchmark`, and the paired statistics with
`python scripts/paired_stats.py`.

Methodology and limitations: [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md).

## 1. Main table

Global model, held-out 20 % test split, final round, mean ± population std over seeds
42, 49, 56, 63, 70. Baselines use the same seeds (identical splits and shards).

| Method | Accuracy | Mean AUC-ROC | ε (δ = 1e-5) |
|---|---|---|---|
| Centralized | 0.846 ± 0.017 | 0.919 | — |
| Local-only (3 nodes, mean) | 0.818 ± 0.019 | — | — |
| Plain FedAvg (no clip, no noise) | 0.830 ± 0.020 | 0.924 | ∞ |
| FedAvg + DP, nm = 1.5 | 0.757 ± 0.058 | 0.849 | 28.6 |
| FedAvg + DP + simulated SMPC, nm = 1.1 | 0.800 ± 0.046 | 0.888 | 43.9 |
| "Full stack", nm = 0.8 | 0.833 ± 0.033 | 0.909 | 73.3 |

Per-seed accuracies are in the JSON (`ablation_full[*].seed_runs[*].final_global_accuracy`).
Each test set holds 61 rows, so one prediction is 1.6 percentage points.

## 2. Paired comparison (output of `scripts/paired_stats.py`)

Mean per-seed accuracy difference, standard error, paired *t* (df = 4, two-sided 5 %
critical |t| = 2.776).

| Condition | vs local-only | vs centralized |
|---|---|---|
| Plain FedAvg | +0.012 (t = 1.51) | −0.016 (t = −2.24) |
| DP nm = 1.5 | −0.060 (t = −2.01) | −0.089 (t = −3.09, **significant**) |
| DP + SMPC nm = 1.1 | −0.018 (t = −0.77) | −0.046 (t = −2.19) |
| Full stack nm = 0.8 | +0.015 (t = 0.84) | −0.013 (t = −0.93) |

## 3. Interpretation — what this does and does not show

1. **No federated condition is significantly better than training locally.** The two federated conditions that edge ahead (+1.2 and +1.5 points) are well inside noise. With five seeds on one 303-row dataset this experiment cannot establish a benefit from federation.
2. **Adding DP noise costs accuracy, as expected,** but the cost is non-monotonic in the conditions because they use different noise multipliers (1.5, 1.1, 0.8) and different SMPC flags. These four rows are not a controlled ablation: SMPC changes nothing numerically (it reproduces the plain sum), so the differences between rows 2–4 are due to the noise level alone.
3. **The privacy guarantees are weak.** ε of 28.6–73.3 at δ = 1e-5 means the formal bound is essentially vacuous for individual-level protection. They are reported so that readers can see the accountant output, not as evidence of strong privacy.
4. **A lower noise multiplier gives a higher ε and higher accuracy** (0.8 → 73.3, 1.5 → 28.6). Reading the "full stack" row as "the best" is therefore misleading: it is the *least private* DP row.
5. **Byzantine test.** With 4 nodes, one scripted attacker injecting large Gaussian noise (mean 5, sd 2) from round 3 is flagged (`Hospital-1`) by the norm-ratio heuristic. This is a single seed and a single easy attack; it says nothing about stealthy attacks. Global accuracy over rounds in that run ranges 0.77–0.85 (`byzantine_resilience.per_round_global_accuracy`) because flagged updates are still aggregated after clipping.
6. **Communication overhead** in the JSON (`communication_overhead_bytes`, 672 bytes per round for every condition) is an **analytical estimate** from parameter counts (`_comm_overhead_bytes`), not a measurement. It applies the same factor 2 to conditions that use no secret sharing. Do not cite it as measured overhead.
7. **No wall-clock, scalability, or on-chain cost results exist.** The benchmark runs in well under a second and touches no blockchain.

## 4. Provenance and earlier numbers

- An earlier version of this repository reported hard-coded or stale numbers; those were removed. An intermediate run of this branch reported centralized 0.869 / local-only 0.842 from a **single seed (42)**. That comparison was unpaired with the 5-seed federated means; the baselines were changed to be computed over the same five seeds, giving the values above. The federated rows did not change.
- `data/ablation_results.json` is produced by `scripts/ablation_study.py` (single seed 42, 10 rounds) and gives different single-run numbers (e.g. plain FedAvg 0.8525, DP nm = 1.5 0.7213). It is a quick illustration, not a replacement for the 5-seed table.
- The JSON field `metadata.dataset` and the table above describe the same data file (SHA-256 recorded in [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md)).

## 5. What would make the results meaningful

More data (larger or multiple datasets, and a real multi-site setting), truly non-IID sharding, a ≥ 30-seed protocol with bootstrap or t-based intervals, an independent DP accountant cross-check, subsampled clients so privacy amplification applies, and a controlled ablation that varies one factor at a time.
