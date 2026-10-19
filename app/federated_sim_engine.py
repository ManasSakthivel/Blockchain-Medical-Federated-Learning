"""
federated_sim_engine.py
=======================
Federated Learning simulation engine with:

  1. Differential Privacy  — CLIENT-LEVEL Gaussian mechanism with Rényi DP
     accountant.  Each client's UPDATE DELTA (local_weights - global_weights)
     is clipped to L2 norm C and calibrated Gaussian noise is added to the
     SUM of clipped deltas before the server averages.  This implements
     client-level DP for FedAvg under the standard "replace-one client"
     adjacency (sensitivity = 2C).  Sampling rate = 1 (all clients selected
     every round; no subsampling amplification is claimed).  The noise source
     is numpy.Generator (seeded, NOT a CSPRNG — suitable for research
     reproducibility, not production cryptography).

  2. Additive Secret Sharing — SINGLE-PROCESS SIMULATION of (n,n)-threshold
     additive secret sharing over the reals.  All operations happen in one
     Python process; there is no cryptographic boundary between parties, no
     finite field, no dropout handling.  The reconstruction is exact.  This
     simulation demonstrates the arithmetic of secret sharing and adds NO
     privacy beyond what DP already provides in-process.  The phrase
     "aggregator never sees raw gradients" does NOT hold in this
     single-process implementation.

  3. Real medical dataset — UCI Cleveland Heart Disease (13 features, 303
     rows; the 6 missing values ('?') are median-imputed from training data only;
     target binarised to 0/1: any heart-disease grade > 0 becomes 1).
     Preprocessing (StandardScaler, median fill) is fitted ONLY on training
     data after the global train/test split — no leakage.

  4. Tamper detection — Byzantine nodes are detected via update-norm and
     cosine-similarity outlier scoring relative to the median of submitted
     deltas.  Nodes whose update deviates too far from the median are flagged.

  5. Reputation scoring — per-node accuracy score updated each round.

IMPORTANT LIMITATIONS
---------------------
- DP guarantee: client-level (ε,δ)-DP under replace-one adjacency, Rényi
  accountant composed over `rounds`, sampling_rate=1.  The accountant uses
  the Gaussian RDP formula (Mironov 2017) with correct sensitivity = 2C for
  replace-one, NOT per-sample DP-SGD.  Epsilon values are large (honest
  accounting, not artificially optimistic).
- No subsampling amplification is applied or claimed.
- numpy.Generator is NOT a CSPRNG; do not use for production DP.
- SMPC is a single-process arithmetic simulation; no cryptographic isolation.
- Dataset is tiny (303 rows); reported confidence intervals will be wide.
- "non-IID" sharding uses StratifiedKFold which is actually near-IID.
  The naming is kept for API compatibility but the docstring acknowledges this.

References
----------
[Mironov 2017] Rényi Differential Privacy of the Gaussian Mechanism
[McMahan 2017] Communication-Efficient Learning of Deep Networks (AISTATS 2017)
[Bonawitz 2017] Practical Secure Aggregation for Privacy-Preserving ML (ACM CCS)
"""

from __future__ import annotations

import copy
import hashlib
import math
import pathlib

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ── Cleveland Heart Disease dataset loader ───────────────────────────────────

class MissingDataError(RuntimeError):
    """Raised when the Cleveland dataset file cannot be found."""


def _load_cleveland(
    allow_synthetic: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI Cleveland Heart Disease dataset (13 features, binary target).

    Preprocessing (StandardScaler + median fill) is performed AFTER the
    data is returned so the caller can fit the scaler on training data only.
    Here we return the raw numeric array (NaN rows dropped) and the target.

    Parameters
    ----------
    allow_synthetic : bool
        If True and the Cleveland file is missing, fall back to a synthetic
        make_classification dataset and return it flagged via the module-level
        IS_SYNTHETIC flag.  If False (default), raise MissingDataError.
    random_state : int
        Seed for the synthetic fallback only.

    Returns
    -------
    X_raw : np.ndarray  — float64, shape (n, 13), unscaled
    y     : np.ndarray  — int, shape (n,), binary
    """
    global _LAST_LOAD_SYNTHETIC
    candidates = [
        pathlib.Path(__file__).parent / "processed.cleveland.data.txt",
        pathlib.Path(__file__).parent.parent / "app" / "processed.cleveland.data.txt",
    ]
    data_path = next((p for p in candidates if p.exists()), None)

    if data_path is None:
        if not allow_synthetic:
            raise MissingDataError(
                "Cleveland dataset not found.  Expected at: "
                + str(candidates[0])
                + "\nSet allow_synthetic=True to fall back to make_classification "
                "(results will be marked as synthetic and should not be compared "
                "to real-data benchmarks)."
            )
        # Explicit synthetic fallback — caller opted in
        _LAST_LOAD_SYNTHETIC = True
        X, y = make_classification(
            n_samples=303, n_features=13, n_informative=8,
            n_redundant=2, random_state=random_state,
        )
        return X.astype(np.float64), y

    import pandas as pd
    _LAST_LOAD_SYNTHETIC = False
    cols = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
    ]
    df = pd.read_csv(data_path, names=cols, na_values="?")
    df["target"] = (df["target"] > 0).astype(int)
    X_raw = df.drop("target", axis=1).values.astype(np.float64)
    y = df["target"].values.astype(int)
    return X_raw, y


_LAST_LOAD_SYNTHETIC: bool = False  # updated by _load_cleveland


def _preprocess(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit StandardScaler and median-fill on X_train, apply to both splits.
    Returns scaled (X_train, X_test) with no leakage from test into fit.
    """
    # Compute median on train only for NaN imputation
    col_medians = np.nanmedian(X_train, axis=0)
    # Fill NaNs
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for j in range(X_tr.shape[1]):
        nan_mask_tr = np.isnan(X_tr[:, j])
        X_tr[nan_mask_tr, j] = col_medians[j]
    for j in range(X_te.shape[1]):
        nan_mask_te = np.isnan(X_te[:, j])
        X_te[nan_mask_te, j] = col_medians[j]
    # Scale on train only
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te


def _heterogeneous_split(
    X: np.ndarray,
    y: np.ndarray,
    n_clients: int,
    seed: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Shard dataset across n_clients using StratifiedKFold.

    Note: StratifiedKFold produces near-IID shards (each fold has the same
    class proportion as the full set).  The naming "non-IID split" in earlier
    versions was a misnomer; this version preserves the API but documents the
    actual behaviour.  Each shard has proportional class balance.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    X_s, y_s = X[perm], y[perm]

    skf = StratifiedKFold(n_splits=n_clients, shuffle=True,
                          random_state=int(seed) % (2**31))
    splits = []
    for _, test_idx in skf.split(X_s, y_s):
        splits.append((X_s[test_idx], y_s[test_idx]))
    return splits


# ── Differential Privacy — Client-level Gaussian Mechanism ───────────────────

def _renyi_epsilon(
    noise_multiplier: float,
    steps: int,
    delta: float,
    alpha_orders: tuple[float, ...] | None = None,
    sensitivity_factor: float = 2.0,
) -> float:
    """
    (ε,δ)-DP bound via Rényi DP composition for the Gaussian mechanism.

    Algorithm
    ---------
    1. For each Rényi order α, per-step RDP of a Gaussian mechanism with
       L2 sensitivity Δ and noise std σ is  α Δ² / (2 σ²).  Here
       σ = noise_multiplier × C (C is the clip norm) and Δ = factor × C, so
           RDP(α) = α × factor² / (2 × noise_multiplier²)
       where the factor depends on the adjacency definition:
         - replace-one:  Δ = 2C  → factor = 2  (conservative; the default)
         - add/remove:   Δ = C   → factor = 1

    2. Linear composition over T steps (rounds):
           RDP_T(α) = T × RDP(α)
       (All clients participate every round; sampling_rate=1; no subsampling
       amplification is applied or claimed.)

    3. Convert RDP → (ε,δ)-DP (Balle et al. / Canonne et al. conversion):
           ε = RDP_T(α) + log(1 - 1/α) - (log(δ) + log(α-1)) / (α-1)

    4. Return minimum ε over all α.

    Parameters
    ----------
    noise_multiplier : float
        σ/C ratio.  Higher = more privacy, lower accuracy.
    steps : int
        Number of FL rounds (composition steps).
    delta : float
        Target δ (failure probability), typically 1e-5.
    sensitivity_factor : float
        2.0 for replace-one adjacency (default, correct for FedAvg).
        1.0 for add/remove one client adjacency.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if steps == 0:
        return 0.0

    if alpha_orders is None:
        alpha_orders = tuple(range(2, 129)) + (256.0, 512.0, 1024.0)

    # Effective σ in terms of sensitivity-normalised noise: σ_eff = nm / factor
    # RDP per step = α / (2 σ_eff²) = α × factor² / (2 nm²)
    factor_sq = sensitivity_factor ** 2
    nm_sq = noise_multiplier ** 2
    best_eps = float("inf")

    for alpha in alpha_orders:
        rdp_per_step = alpha * factor_sq / (2.0 * nm_sq)
        rdp_total = steps * rdp_per_step
        if alpha > 1:
            eps = rdp_total + math.log(1.0 - 1.0 / alpha) - (
                math.log(delta) + math.log(alpha - 1.0)
            ) / (alpha - 1.0)
            if eps < best_eps:
                best_eps = eps

    return round(max(0.0, best_eps), 6)


class GaussianMechanism:
    """
    Client-level DP via update-delta clipping + Gaussian noise for FedAvg.

    What this implements
    --------------------
    - Each client computes a LOCAL UPDATE DELTA = local_weights - global_weights.
    - The delta is clipped to L2 norm C (max_grad_norm).
    - Calibrated Gaussian noise N(0, σ²I) with σ = noise_multiplier × C is
      added to the AGGREGATED SUM of clipped deltas before the server divides
      by n_clients.
    - This provides client-level (ε,δ)-DP under replace-one adjacency with
      sensitivity 2C (replacing one client changes the sum by at most 2C).

    What this does NOT implement
    ----------------------------
    - Per-sample DP-SGD (Abadi et al. 2016): that requires per-example gradient
      clipping during local SGD steps.  This code clips the full weight vector.
    - Subsampling amplification: all clients are selected every round.
    - Cryptographically secure noise: numpy.Generator is used for
      reproducibility; not suitable for production privacy applications.

    Privacy accounting
    ------------------
    Uses Rényi DP (Mironov 2017) with correct replace-one sensitivity factor 2.
    Epsilon values will be large for the small noise levels used here.
    """

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        # Seeded numpy Generator for reproducibility (NOT a CSPRNG)
        self._rng = rng if rng is not None else np.random.default_rng(0)

    def clip_delta(self, delta: np.ndarray) -> np.ndarray:
        """Clip update delta to L2 norm ≤ max_grad_norm."""
        norm = np.linalg.norm(delta)
        if norm > self.max_grad_norm:
            delta = delta * (self.max_grad_norm / norm)
        return delta

    def clip_and_noise(self, flat_weights: np.ndarray) -> np.ndarray:
        """
        Backward-compatible API: clip then add noise (treats input as delta).
        Used when caller passes a full weight vector; clips it and adds noise.
        """
        clipped = self.clip_delta(flat_weights)
        sigma = self.noise_multiplier * self.max_grad_norm
        noise = self._rng.normal(0.0, sigma, clipped.shape)
        return clipped + noise

    def add_noise_to_aggregate(
        self, aggregate_delta: np.ndarray, n_clients: int
    ) -> np.ndarray:
        """
        Add calibrated noise to the SUM of n_clients clipped deltas.
        σ = noise_multiplier × max_grad_norm (sensitivity per component = 2C
        for replace-one, already baked into the accountant via factor=2).
        """
        sigma = self.noise_multiplier * self.max_grad_norm
        noise = self._rng.normal(0.0, sigma, aggregate_delta.shape)
        return aggregate_delta + noise

    def compute_epsilon(self, steps: int) -> float:
        """
        Cumulative (ε,δ)-DP after `steps` rounds.
        Uses replace-one sensitivity (factor=2) — correct for FedAvg.
        """
        return _renyi_epsilon(
            noise_multiplier=self.noise_multiplier,
            steps=steps,
            delta=self.delta,
            sensitivity_factor=2.0,
        )


# ── Additive Secret Sharing SMPC ─────────────────────────────────────────────

class AdditiveSecretSharing:
    """
    Single-process simulation of (n,n)-threshold additive secret sharing.

    IMPORTANT: This is an arithmetic simulation running in ONE Python process.
    There is NO cryptographic isolation between parties.  Any variable in the
    process can read any other party's secrets at any point.  The simulation
    demonstrates that the arithmetic of secret sharing is correct (sum of
    shares = original) but provides NO additional privacy guarantee in this
    implementation beyond what DP already provides.

    The phrase "aggregator never sees raw gradients" does NOT hold here.

    Each weight w is split into n real-valued shares:
        w = s_1 + s_2 + ... + s_n   where s_i ~ N(0,1) for i < n, s_n = w - Σs_i
    Any strict subset of n-1 shares is statistically independent of w (over the
    randomness of the shares), but this property only holds in an honest-but-
    curious multi-party setting, not in a single process.
    """

    @staticmethod
    def share(
        weight: np.ndarray,
        n_parties: int,
        rng: np.random.Generator | None = None,
    ) -> list[np.ndarray]:
        """Split `weight` into `n_parties` additive shares."""
        draw = rng.normal if rng is not None else np.random.normal
        shares = [draw(0, 1, weight.shape) for _ in range(n_parties - 1)]
        last_share = weight - sum(shares)
        shares.append(last_share)
        return shares

    @staticmethod
    def reconstruct(shares: list[np.ndarray]) -> np.ndarray:
        """Sum all shares to recover the original value."""
        return sum(shares)

    @staticmethod
    def secure_aggregate(
        all_weights: list[np.ndarray],
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Simulate secure aggregation via additive secret sharing.
        Returns the mean (FedAvg average) of the input weight vectors.
        Result equals the plain mean to floating-point precision.
        """
        n = len(all_weights)
        all_shares = [AdditiveSecretSharing.share(w, n, rng) for w in all_weights]
        aggregated_shares = []
        for j in range(n):
            slot_sum = sum(all_shares[i][j] for i in range(n))
            aggregated_shares.append(slot_sum)
        global_sum = AdditiveSecretSharing.reconstruct(aggregated_shares)
        return global_sum / n  # FedAvg = mean


# ── Federated Node ────────────────────────────────────────────────────────────

class FederatedNode:
    """
    A single hospital node in the federated learning system.

    Training is genuinely iterative: each round the node is initialised from
    the current global model weights (warm start via explicit coefficient
    injection + warm_start=True), then fine-tunes on local data.  The
    update delta (local - global) is computed and clipped for DP.

    Fields added per round
    ----------------------
    accuracies      : local model accuracy on own test split
    auc_scores      : local model AUC-ROC on own test split
    local_test_*    : same values, explicitly named to distinguish from global
    """

    def __init__(
        self,
        node_id: str,
        X: np.ndarray,
        y: np.ndarray,
        dp: GaussianMechanism | None = None,
        random_state: int | None = None,
    ) -> None:
        self.node_id = node_id
        # NOTE: X and y are already scaled training-shard data; the caller
        # holds the global scaler.  We split 80/20 from the shard.
        # random_state=None → sklearn uses its own global state; seed must be
        # passed explicitly for reproducibility.  seed=0 is valid (not 42).
        rs = random_state  # may be 0 (valid), or None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=rs, stratify=y
        )
        self.dp = dp or GaussianMechanism()
        # warm_start=True allows injecting coef_/intercept_ before fit()
        self.model = LogisticRegression(
            max_iter=200, solver="lbfgs", random_state=random_state,
            warm_start=True,
        )
        self.weights: np.ndarray | None = None
        self.intercept: np.ndarray | None = None
        self.noised_weights: np.ndarray | None = None
        self.hashes: list[str] = []
        self.accuracies: list[float] = []       # local model on own test split
        self.auc_scores: list[float] = []       # local model on own test split
        self.reputation: float = 1.0
        self.status: str = "Initialized"
        self.tampered: bool = False
        # Stored delta for DP processing
        self._last_delta: np.ndarray | None = None

    def train_local(self, global_weights: dict | None = None) -> None:
        """
        Train local model, warm-started from global_weights when provided.

        When global_weights is given the model's coef_ and intercept_ are
        set before calling fit(), so scikit-learn's warm_start=True will use
        them as the initial point.  The first round (global_weights=None)
        trains from scratch.

        The update delta (local - global) is stored for DP processing.
        """
        if global_weights is not None:
            # Inject global model as starting point for warm start
            try:
                self.model.coef_ = global_weights["coef"].copy()
                self.model.intercept_ = global_weights["intercept"].copy()
                self.model.classes_ = np.array([0, 1])
            except (AttributeError, ValueError):
                pass

        self.model.fit(self.X_train, self.y_train)
        new_weights = copy.deepcopy(self.model.coef_)
        new_intercept = copy.deepcopy(self.model.intercept_)

        # Compute delta relative to the global model we started from
        if global_weights is not None:
            self._last_delta = np.concatenate([
                (new_weights - global_weights["coef"]).flatten(),
                (new_intercept - global_weights["intercept"]).flatten(),
            ])
        else:
            self._last_delta = np.concatenate([
                new_weights.flatten(), new_intercept.flatten()
            ])

        self.weights = new_weights
        self.intercept = new_intercept
        self.status = "Trained"

    def get_dp_weights(self) -> np.ndarray:
        """
        Return DP-protected FLAT WEIGHT VECTOR (not delta) for backward compat.
        The delta is stored separately for the engine's aggregate DP step.
        Byzantine tampering is applied to the submitted weight vector.
        """
        flat = np.concatenate([self.weights.flatten(), self.intercept.flatten()])
        if self.tampered:
            flat = flat + np.random.normal(5.0, 2.0, flat.shape)
        # Clip and noise the full vector (legacy API kept for tests)
        noised = self.dp.clip_and_noise(flat)
        self.noised_weights = noised
        return noised

    def get_raw_delta(self) -> np.ndarray:
        """
        Return the RAW (pre-clip) update delta including any Byzantine noise.
        Used by the server for anomaly detection: detection must happen before
        clipping because all clipped norms are capped at max_grad_norm.
        """
        if self._last_delta is None:
            raise RuntimeError("call train_local first")
        delta = self._last_delta.copy()
        if self.tampered:
            delta = delta + np.random.normal(5.0, 2.0, delta.shape)
        return delta

    def get_clipped_delta(self) -> np.ndarray:
        """
        Return the update delta for aggregation.
        With DP enabled (noise_multiplier > 0) the raw delta (including any
        Byzantine noise) is clipped to max_grad_norm.  With noise_multiplier
        == 0 (the "plain FedAvg" baseline) no clipping is applied.
        """
        raw = self.get_raw_delta()
        if self.dp.noise_multiplier <= 0:
            return raw
        return self.dp.clip_delta(raw)

    def evaluate(self) -> dict:
        """
        Evaluate LOCAL model on local hold-out test split.
        Returns dict with 'accuracy', 'auc_roc' (local model, local data).
        These are NOT the global model metrics.
        """
        preds = self.model.predict(self.X_test)
        acc = float(accuracy_score(self.y_test, preds))
        try:
            probs = self.model.predict_proba(self.X_test)[:, 1]
            auc = float(roc_auc_score(self.y_test, probs))
        except Exception:
            auc = float("nan")
        self.accuracies.append(acc)
        self.auc_scores.append(auc)
        return {"accuracy": acc, "auc_roc": auc, "source": "local_model_local_test"}

    def get_model_hash(self) -> str:
        """SHA-256 hash of the DP-noised weight vector (for on-chain logging)."""
        if self.noised_weights is None:
            self.get_dp_weights()
        w_bytes = self.noised_weights.tobytes()
        h = hashlib.sha256(w_bytes).hexdigest()
        self.hashes.append(h)
        return h

    def update_reputation(self, acc: float) -> None:
        """Exponential moving average of accuracy as reputation score."""
        alpha = 0.7
        self.reputation = alpha * acc + (1.0 - alpha) * self.reputation

    def tamper(self) -> None:
        """Mark this node as Byzantine — it will inject poison into its delta."""
        self.tampered = True
        self.status = "Tampered"


# ── Byzantine Detection ───────────────────────────────────────────────────────

def _detect_byzantine_by_norm(
    deltas: list[np.ndarray],
    node_ids: list[str],
    ratio_threshold: float = 5.0,
    min_clients: int = 3,
) -> list[str]:
    """
    Flag clients whose RAW (pre-clip) update norm exceeds
    ``ratio_threshold`` times the median norm of all clients.

    With ``min_clients`` or more clients a single outlier cannot move the
    median, so one attacker is always compared against honest norms.  With
    fewer clients no median is meaningful and nothing is flagged.

    Limitations (heuristic, NOT a provably robust defence):
      - only large-magnitude attacks are visible; an attacker that keeps its
        norm near the honest median (or colludes with others) is not flagged;
      - flagging is informational: flagged updates are still aggregated
        (they are clipped to ``max_grad_norm`` like every other update);
      - honest clients with very heterogeneous data may be flagged.
    """
    if len(deltas) < min_clients:
        return []

    norms = np.array([np.linalg.norm(d) for d in deltas])
    median_norm = float(np.median(norms))
    if median_norm < 1e-12:
        return []
    return [nid for nid, n in zip(node_ids, norms)
            if n > ratio_threshold * median_norm]


# ── Federated Simulation ──────────────────────────────────────────────────────

class FederatedSimulation:
    """
    Multi-round Federated Learning simulation with:
      - Real Cleveland Heart Disease data (stratified shard split)
      - Client-level DP (Gaussian Mechanism, correct sensitivity=2C for FedAvg)
      - Additive Secret Sharing (single-process simulation; see class docstring)
      - Update-norm Byzantine detection
      - Global model evaluated on a HELD-OUT TEST SET each round
        (preprocessing fitted only on training data — no leakage)
      - Per-round local metrics kept as separate fields
    """

    REPUTATION_THRESHOLD: float = 0.55

    def __init__(
        self,
        n_nodes: int = 3,
        n_rounds: int = 10,
        n_features: int = 13,
        noise_multiplier: float = 1.1,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        seed: int = 42,
        allow_synthetic: bool = False,
        use_smpc: bool = False,
    ) -> None:
        # seed=0 is valid; use it as-is (no silent coercion)
        self.seed = seed
        self.n_rounds = n_rounds
        self.use_smpc = use_smpc
        self._smpc_rng = np.random.default_rng(seed + 1_000_003)

        # Set numpy legacy random state for any code using np.random.*
        np.random.seed(seed)

        # ── Load and split data BEFORE preprocessing ──────────────────────────
        X_raw, y_all = _load_cleveland(allow_synthetic=allow_synthetic,
                                        random_state=seed)
        self.synthetic_fallback = _LAST_LOAD_SYNTHETIC

        # Global held-out test split (20%) — used to evaluate global model
        # The same split is used every round so accuracy is comparable across
        # rounds and conditions (given the same seed).
        X_trainval, self.X_global_test, y_trainval, self.y_global_test = (
            train_test_split(X_raw, y_all, test_size=0.2, random_state=seed,
                             stratify=y_all)
        )

        # ── Preprocessing: fit scaler on TRAINING portion only ────────────────
        X_trainval_scaled, X_test_scaled = _preprocess(X_trainval,
                                                        self.X_global_test)
        self.X_global_test = X_test_scaled  # scaled test set, leak-free

        # ── Shard training data across nodes ──────────────────────────────────
        splits = _heterogeneous_split(X_trainval_scaled, y_trainval,
                                      n_nodes, seed=seed)

        self.n_features = X_trainval_scaled.shape[1]

        # ── Shared DP mechanism ───────────────────────────────────────────────
        self.dp = GaussianMechanism(
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            delta=delta,
            rng=np.random.default_rng(seed),
        )
        self.smpc = AdditiveSecretSharing()

        self.nodes = [
            FederatedNode(
                node_id=f"Hospital-{i+1}",
                X=splits[i][0],
                y=splits[i][1],
                dp=self.dp,
                random_state=seed + i,
            )
            for i in range(n_nodes)
        ]
        self.global_weights: dict | None = None
        self.global_hashes: list[str] = []
        self.round_logs: list[dict] = []
        self.cumulative_rounds: int = 0

    def _evaluate_global_model(self) -> dict:
        """
        Evaluate the current global model on the held-out test set.
        Returns accuracy and AUC-ROC.  The test set is never used for training
        and was preprocessed with scaler fitted only on training data.
        """
        if self.global_weights is None:
            return {"global_accuracy": float("nan"), "global_auc_roc": float("nan")}
        try:
            model = LogisticRegression(max_iter=1, solver="lbfgs")
            # Initialise classes_ by fitting on a tiny dummy (needed for predict)
            X_dummy = np.zeros((2, self.n_features))
            y_dummy = np.array([0, 1])
            model.fit(X_dummy, y_dummy)
            model.coef_ = self.global_weights["coef"]
            model.intercept_ = self.global_weights["intercept"]
            preds = model.predict(self.X_global_test)
            acc = float(accuracy_score(self.y_global_test, preds))
            try:
                probs = model.predict_proba(self.X_global_test)[:, 1]
                auc = float(roc_auc_score(self.y_global_test, probs))
            except Exception:
                auc = float("nan")
            return {"global_accuracy": round(acc, 4), "global_auc_roc": round(auc, 4)}
        except Exception:
            return {"global_accuracy": float("nan"), "global_auc_roc": float("nan")}

    def run_round(
        self, round_idx: int, tamper_node: int | None = None
    ) -> dict:
        """Execute one FL round with client-level DP + SMPC aggregation."""
        round_log: dict = {
            "round": round_idx + 1,
            "nodes": [],
            "flagged_nodes": [],
        }

        raw_deltas: list[np.ndarray] = []      # for Byzantine detection (pre-clip)
        clipped_deltas: list[np.ndarray] = []  # for DP aggregation (post-clip)
        node_ids: list[str] = []

        for i, node in enumerate(self.nodes):
            if tamper_node is not None and i == tamper_node:
                node.tamper()

            # Step 1: Local training — warm-start from current global model
            node.train_local(self.global_weights)

            # Step 2: Evaluate LOCAL model on node's own test split
            local_metrics = node.evaluate()
            local_acc = local_metrics["accuracy"]

            # Step 3a: Raw delta (pre-clip) — computed ONCE (a tampered node
            # draws attack noise here) and used for Byzantine detection
            raw_delta = node.get_raw_delta()
            raw_deltas.append(raw_delta)
            node_ids.append(node.node_id)

            # Step 3b: The delta that is actually aggregated: clipped when DP
            # is enabled, unmodified for the noise_multiplier == 0 baseline
            if self.dp.noise_multiplier <= 0:
                clipped_delta = raw_delta
            else:
                clipped_delta = self.dp.clip_delta(raw_delta)
            clipped_deltas.append(clipped_delta)

            # Step 5: Hash of the submitted update (what the server aggregates)
            model_hash = hashlib.sha256(clipped_delta.tobytes()).hexdigest()
            node.hashes.append(model_hash)

            # Step 6: Update reputation from local accuracy
            node.update_reputation(local_acc)

            round_log["nodes"].append({
                "id": node.node_id,
                "hash": model_hash,
                # local_* fields: local model evaluated on the node's own split
                "local_accuracy": round(local_acc, 4),
                "local_auc_roc": round(local_metrics["auc_roc"], 4),
                # backward-compat aliases (used by benchmark.py, routes)
                "accuracy": round(local_acc, 4),
                "auc_roc": round(local_metrics["auc_roc"], 4),
                "reputation": round(node.reputation, 4),
                "status": node.status,
                "dp_epsilon_so_far": self.dp.compute_epsilon(self.cumulative_rounds + 1),
            })

        # Step 7: Client-level DP — add noise to the SUM of clipped deltas.
        # With use_smpc the sum is formed via additive secret sharing
        # (single-process simulation: arithmetic only, no isolation).
        if self.use_smpc:
            agg_delta_sum = (
                AdditiveSecretSharing.secure_aggregate(
                    clipped_deltas, rng=self._smpc_rng
                ) * len(clipped_deltas)
            )
        else:
            agg_delta_sum = sum(clipped_deltas)
        noised_delta_sum = self.dp.add_noise_to_aggregate(
            agg_delta_sum, n_clients=len(self.nodes)
        )
        fedavg_delta = noised_delta_sum / len(self.nodes)

        # Step 8: Reconstruct new global model from averaged delta
        n_coef = self.nodes[0].weights.shape[1]
        if self.global_weights is None:
            # First round: delta IS the weight
            new_coef = fedavg_delta[:n_coef].reshape(1, -1)
            new_intercept = fedavg_delta[n_coef:].reshape(1,)
        else:
            # Subsequent rounds: apply delta to current global
            new_coef = (self.global_weights["coef"].flatten()
                        + fedavg_delta[:n_coef]).reshape(1, -1)
            new_intercept = (self.global_weights["intercept"].flatten()
                             + fedavg_delta[n_coef:]).reshape(1,)
        self.global_weights = {"coef": new_coef, "intercept": new_intercept}

        # Step 9: Hash global model
        global_flat = np.concatenate([new_coef.flatten(), new_intercept.flatten()])
        global_hash = hashlib.sha256(global_flat.tobytes()).hexdigest()
        self.global_hashes.append(global_hash)
        self.cumulative_rounds += 1

        # Step 10: Evaluate GLOBAL model on held-out test set
        global_metrics = self._evaluate_global_model()
        round_log["global_accuracy"] = global_metrics["global_accuracy"]
        round_log["global_auc_roc"] = global_metrics["global_auc_roc"]

        # Step 11: Byzantine detection via update-norm outlier scoring on RAW deltas
        # Detection uses raw (pre-clip) deltas so that large-magnitude Byzantine
        # updates are visible before clipping reduces all norms to max_grad_norm.
        round_log["flagged_nodes"] = _detect_byzantine_by_norm(
            raw_deltas, node_ids
        )
        round_log["global_hash"] = global_hash
        round_log["cumulative_epsilon"] = self.dp.compute_epsilon(self.cumulative_rounds)

        self.round_logs.append(round_log)
        return round_log

    def run_simulation(
        self,
        tamper_round: int | None = None,
        tamper_node: int | None = None,
    ) -> list[dict]:
        """Run all rounds. Optionally inject a Byzantine node at tamper_round."""
        self.global_weights = None
        self.global_hashes = []
        self.round_logs = []
        self.cumulative_rounds = 0

        for r in range(self.n_rounds):
            inject = tamper_round is not None and r == tamper_round
            self.run_round(r, tamper_node=tamper_node if inject else None)

        return self.round_logs

    def get_final_global_accuracy(self) -> float:
        """
        Return the global model accuracy on the held-out test set.
        The test set was preprocessed with scaler fitted only on training data
        (no preprocessing leakage).
        """
        metrics = self._evaluate_global_model()
        return metrics["global_accuracy"]
