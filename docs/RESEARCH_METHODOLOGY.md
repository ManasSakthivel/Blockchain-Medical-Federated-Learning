# Research methodology

This describes what `app/federated_sim_engine.py` and `app/benchmark.py` actually
do. Nothing here is a claim about a deployed system: everything runs in **one
Python process** on a public dataset.

## 1. Data

- **UCI Cleveland Heart Disease (processed)**, `app/processed.cleveland.data.txt`: 303 rows, 13 features, SHA-256 `a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8`.
- Target: original grade 0–4 binarised as `grade > 0 → 1` (`app/federated_sim_engine.py:134`).
- Missing values: 6 `?` entries (`ca`, `thal`), median-imputed using **training-set medians only** (`_preprocess`, line 144).
- Provenance: the file was committed by the repository author. It matches the format of the UCI "processed.cleveland.data" file, but it has **not** been byte-compared to a fresh UCI download in this audit.
- If the file is missing the loader raises `MissingDataError`; it only falls back to synthetic data when explicitly asked (`allow_synthetic=True`), and then the result is flagged `synthetic_fallback`.
- 303 rows is very small. Any accuracy difference below a few percentage points is within noise (see §5).

## 2. Split and leakage controls

1. Stratified 80/20 global split (`train_test_split(..., stratify=y)`, `random_state=seed`) → a **held-out global test set** never used for training or preprocessing.
2. `StandardScaler` and median-fill are fitted on the 80 % training part only, then applied to the test set (`_preprocess`).
3. The 80 % training part is sharded across nodes with `StratifiedKFold(shuffle=True)`. This yields **near-IID** shards; the function is still named `_heterogeneous_split` for compatibility. **No non-IID (label- or feature-skew) experiment is run**, so claims about heterogeneity are unsupported.
4. Each node additionally holds out a local split to compute `local_accuracy` / reputation; reported headline accuracy is the **global model on the global test set**.

## 3. Algorithms

**Model.** `sklearn.linear_model.LogisticRegression` (lbfgs, `max_iter=200` per node-round). Each node warm-starts from the current global weights.

**Round (`FederatedSimulation.run_round`).**
1. Each node trains locally and computes the raw update `Δ = w_local − w_global`.
2. With DP enabled (`noise_multiplier > 0`) each `Δ` is clipped to L2 norm `C = 1.0`. With `noise_multiplier = 0` (the "plain FedAvg" baseline) **no clipping and no noise** are applied.
3. The clipped deltas are summed. With `use_smpc=True` the sum is formed with additive secret sharing (§4); otherwise a plain sum.
4. Gaussian noise `N(0, (noise_multiplier·C)² I)` is added to the **sum**, which is then divided by the number of nodes and applied to the global model.
5. The global model is evaluated on the held-out test set; its SHA-256 and each node's update hash are recorded in the round log (not written to any chain).

**Reputation.** Per-node EMA of *local* accuracy (`alpha = 0.7`, `update_reputation`). It is a monitoring value only; it does not weight or exclude updates.

## 4. Privacy mechanisms and their real boundaries

**Differential privacy.** Client-level Gaussian mechanism, replace-one adjacency, sensitivity `2C`, no subsampling (all nodes every round). ε is computed with a Rényi-DP accountant over `T` rounds (`_renyi_epsilon`, lines 198–264): per-step RDP `α·(2)²/(2·nm²)`, composed linearly, converted to (ε, δ=1e-5) and minimised over α ∈ {2…128, 256, 512, 1024}.

Boundaries:
- Local training itself is **not** noised; only the aggregate is. The guarantee therefore assumes the party adding noise is trusted. With one process, that party is the simulator.
- Noise is drawn from `numpy.random.default_rng(seed)`: seeded, deterministic and **not cryptographically secure**. Fine for reproducible science, unusable as a production DP source.
- With 3 clients the resulting ε is large (see §5); no useful privacy claim can be made.
- The accountant covers the mechanism as coded. It has been unit-tested for monotonicity and the `nm = 0 → ∞` case but has **not** been cross-checked against an independent library (e.g. Opacus/TF-Privacy) in this audit.

**"SMPC".** `AdditiveSecretSharing` splits each delta into random additive shares and reconstructs the sum, so the *arithmetic* of secure aggregation is exercised and verified (sum equals plain sum within float tolerance). All shares live in the same process. It provides **no** confidentiality against that process or its operator and must not be described as "the aggregator never sees raw updates". There are no separate parties, no network, no dropout handling, no authenticated channels, and no malicious-security guarantees.

**Byzantine detection.** A node is flagged if its raw-delta norm exceeds `5×` the median norm of all nodes, and only when ≥ 3 nodes participate (`_detect_byzantine_by_norm`, line 580). It is **informational**: flagged updates are still aggregated (after clipping). It detects only large-magnitude attacks; a norm-matched, colluding or targeted attacker evades it, and heterogeneous honest nodes may be flagged. The benchmark's single "poisoning detected: True" result is one attack type at one magnitude with one seed.

## 5. Experimental protocol

Command: `python -m app.benchmark --nodes 3 --rounds 10 --seed 42 --n-seeds 5`. Seeds are `42 + 7i`. For each seed, per condition:

| Condition | noise multiplier | SMPC |
|---|---|---|
| Plain FedAvg | 0 (no clip, no noise) | no |
| FedAvg + DP | 1.5 | no |
| FedAvg + DP + SMPC | 1.1 | yes |
| Full stack | 0.8 | yes |

Baselines (`app/benchmark.py`, `_centralized_baseline`, `_local_only_baseline`), run with the **same seeds** as the federated conditions so every comparison is paired on identical train/test splits: **centralized** logistic regression on the pooled 80 % training split, and **local-only** (each node trains alone on 80 % of its own shard; mean test accuracy of those local models on the global held-out set). Metrics: test accuracy and AUC-ROC on the global held-out test set; ε at the end of training. Reported: mean, standard deviation (population, `np.std`) and a normal-approximation 95 % CI (`1.96·std/√n`) over 5 seeds. With n = 5 the normal approximation understates the interval (a t-interval would be ~40 % wider).

Important design notes for interpreting results:
- The four conditions differ in **both** noise level and SMPC flag, and SMPC does not change the numeric result beyond float rounding. They are therefore *not* a clean ablation of SMPC; the noise multipliers were chosen a priori (not tuned to improve metrics) and are not monotonically ordered with the conditions (1.5 → 1.1 → 0.8).
- The 5 seeds vary the data split, sharding and noise, but there is a single dataset of 303 rows. The CIs do not reflect dataset-level uncertainty.
- No hyper-parameter search was run; no result was selected or discarded after the fact.

## 6. Validity threats

- Tiny dataset; high variance; near-IID shards; one dataset; one model class.
- Local-only baseline matches or exceeds the federated variants, i.e. federation brings no demonstrated benefit here.
- DP guarantee depends on a trusted noise-adding party and a non-CSPRNG.
- Simulation, not deployment: no network, no real institutions, no heterogeneous feature spaces, no adversarial participants beyond the single scripted attacker.
- Earlier versions of this repository's README attributed further capabilities to the source paper (zk-STARK-equivalent proofs, MIMIC-III evaluation, layer-2 scalability, staking/slashing, GDPR compliance). **None of these are implemented or evaluated here**, and the paper PDFs in `docs/` have not been reviewed as part of this work; do not treat them as describing this code.
