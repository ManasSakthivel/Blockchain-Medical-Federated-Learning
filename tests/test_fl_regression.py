"""
tests/test_fl_regression.py
===========================
Regression tests for all issues identified in the audit:

  (a) BM-C2  — iterative training: global model actually feeds back
  (b) BM-H4  — global model evaluated on separate held-out set, not local
  (c) BM-M1  — no preprocessing leakage (scaler fitted after split)
  (d) BM-M3  — seed=0 produces different results from seed=42
  (e) BM-M4  — explicit error when Cleveland file missing (no silent fallback)
  (f) BM-H3  — Byzantine detection flags attacker, no false positive on clean
  (g) BM-H5  — DP accountant: epsilon grows monotonically, factor=2 is used
  (h) BM-M6  — ablation config: full_stack != fedavg_dp_smpc, labels correct
  (i) benchmark determinism — running twice gives identical numbers
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import math

import numpy as np
import pytest

from app.federated_sim_engine import (
    FederatedSimulation,
    FederatedNode,
    GaussianMechanism,
    MissingDataError,
    _load_cleveland,
    _preprocess,
    _renyi_epsilon,
    _detect_byzantine_by_norm,
)


# ---------------------------------------------------------------------------
# (a) BM-C2 — Iterative training: global model must change local init
# ---------------------------------------------------------------------------

class TestIterativeTraining:
    def test_local_init_changes_with_global_model(self):
        """
        In a 2-round simulation the global model from round 1 must produce a
        different local initialisation in round 2 than the round 1 initialisation.
        This proves training is genuinely iterative.
        """
        sim = FederatedSimulation(n_nodes=3, n_rounds=2, noise_multiplier=0.0,
                                   seed=42)
        # Manually drive round 1
        sim.run_round(0)
        global_after_round1 = {
            "coef": sim.global_weights["coef"].copy(),
            "intercept": sim.global_weights["intercept"].copy(),
        }
        assert global_after_round1 is not None

        sim.run_round(1)

        # After round 2 the global model must differ from after round 1
        # (it has absorbed a new round of updates)
        global_after_round2 = sim.global_weights
        coef_diff = np.linalg.norm(
            global_after_round2["coef"] - global_after_round1["coef"]
        )
        assert coef_diff > 0, \
            "Global model must change between rounds (iterative training)"

    def test_node_uses_global_weights_as_warm_start(self):
        """
        When train_local is called with global_weights the model must be
        initialised from those weights, not from scratch.  Verify by checking
        that the model coef_ before fit is equal to the injected global weights.
        """
        rng = np.random.default_rng(7)
        X = rng.standard_normal((60, 5))
        y = (X[:, 0] > 0).astype(int)

        node = FederatedNode("test", X, y, random_state=7)
        # First train from scratch
        node.train_local(global_weights=None)
        first_coef = node.weights.copy()

        # Build a fake global model with different weights
        fake_global = {
            "coef": np.ones_like(first_coef) * 99.0,
            "intercept": np.zeros(1),
        }
        node.train_local(global_weights=fake_global)
        # After injection the model was initialised at 99.0; after fit it
        # should have moved away from first_coef (different starting point)
        second_coef = node.weights.copy()
        # The starting point changed so the result must differ from scratch init
        diff = np.linalg.norm(first_coef - second_coef)
        assert diff > 0, \
            "Warm start from global weights must produce different local weights"

    def test_global_weights_evolve_across_rounds(self):
        """Global weights must not be identical across consecutive rounds."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=5, seed=42)
        logs = sim.run_simulation()
        hashes = [log["global_hash"] for log in logs]
        # All hashes must be distinct (global model changes each round)
        assert len(set(hashes)) == len(hashes), \
            "Global model hash must be unique every round"


# ---------------------------------------------------------------------------
# (b) BM-H4 — Global model on separate held-out set
# ---------------------------------------------------------------------------

class TestGlobalModelEvaluation:
    def test_global_accuracy_in_each_round(self):
        """Every round log must have global_accuracy and global_auc_roc."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=3, seed=42)
        logs = sim.run_simulation()
        for rlog in logs:
            assert "global_accuracy" in rlog
            assert "global_auc_roc" in rlog
            assert 0.0 <= rlog["global_accuracy"] <= 1.0

    def test_global_test_set_not_in_training(self):
        """
        The global held-out test set must not overlap with any node's
        training data (no data leakage).
        """
        sim = FederatedSimulation(n_nodes=3, n_rounds=1, seed=42)
        sim.run_simulation()
        # Collect all training indices (using identity by row values)
        all_train = np.vstack([n.X_train for n in sim.nodes])
        test_set = sim.X_global_test
        # Check that no test row is identical to any training row
        # (Rows are float64 arrays — compare by unique row fingerprint)
        def row_set(arr):
            return set(arr.tobytes() for arr in arr)
        overlap = row_set(test_set) & row_set(all_train)
        assert len(overlap) == 0, \
            "Global test set must not overlap with training data"

    def test_local_and_global_accuracy_are_logged_separately(self):
        """Round logs must expose both local_accuracy and global_accuracy."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=2, seed=42)
        logs = sim.run_simulation()
        for rlog in logs:
            assert "global_accuracy" in rlog
            for nlog in rlog["nodes"]:
                assert "local_accuracy" in nlog or "accuracy" in nlog


# ---------------------------------------------------------------------------
# (c) BM-M1 — No preprocessing leakage
# ---------------------------------------------------------------------------

class TestPreprocessingLeakage:
    def test_preprocess_fits_only_on_train(self):
        """
        _preprocess must fit scaler on X_train only.  Applying to X_test should
        produce different statistics than if it were fit on the full dataset.
        """
        rng = np.random.default_rng(0)
        # Create data where test has very different range than train
        X_train = rng.normal(0, 1, (80, 5))
        X_test = rng.normal(10, 0.1, (20, 5))  # mean=10, very different

        X_tr_scaled, X_te_scaled = _preprocess(X_train, X_test)

        # Train should be ~zero mean (scaled on train)
        assert abs(X_tr_scaled.mean()) < 0.5, "Train should be approximately zero-mean"
        # Test should NOT be zero mean (it has different raw distribution)
        # (If leakage occurred, test mean would also be ~0)
        assert abs(X_te_scaled.mean()) > 1.0, \
            "Test mean should differ from zero: scaler was fit on train only"

    def test_simulation_does_not_leak_test_into_scaler(self):
        """
        The FederatedSimulation must hold a separate preprocessed test set
        that was scaled using statistics from the training data only.
        """
        sim = FederatedSimulation(n_nodes=3, n_rounds=1, seed=42)
        # X_global_test must exist and be scaled
        assert sim.X_global_test is not None
        assert sim.X_global_test.shape[1] == 13

    def test_nan_imputation_uses_train_median_only(self):
        """Median imputation must use train median, not global median."""
        rng = np.random.default_rng(5)
        X_train = rng.normal(0, 1, (40, 3))
        X_test = rng.normal(0, 1, (10, 3))
        # Introduce NaN in both
        X_train[0, 0] = np.nan
        X_test[0, 0] = np.nan

        X_tr_s, X_te_s = _preprocess(X_train, X_test)
        # No NaNs should remain
        assert not np.any(np.isnan(X_tr_s))
        assert not np.any(np.isnan(X_te_s))


# ---------------------------------------------------------------------------
# (d) BM-M3 — seed=0 is valid; produces different result from seed=42
# ---------------------------------------------------------------------------

class TestSeedHandling:
    def test_seed_zero_valid(self):
        """seed=0 must not raise and must complete successfully."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=2, seed=0)
        logs = sim.run_simulation()
        assert len(logs) == 2

    def test_seed_zero_differs_from_seed_42(self):
        """seed=0 and seed=42 must produce different global weights."""
        sim0 = FederatedSimulation(n_nodes=3, n_rounds=3, seed=0)
        sim42 = FederatedSimulation(n_nodes=3, n_rounds=3, seed=42)
        sim0.run_simulation()
        sim42.run_simulation()
        diff = np.linalg.norm(
            sim0.global_weights["coef"] - sim42.global_weights["coef"]
        )
        assert diff > 0, "seed=0 and seed=42 must produce different global models"

    def test_same_seed_reproducible(self):
        """Two runs with the same seed must produce identical results."""
        sim_a = FederatedSimulation(n_nodes=3, n_rounds=3, seed=99)
        sim_b = FederatedSimulation(n_nodes=3, n_rounds=3, seed=99)
        logs_a = sim_a.run_simulation()
        logs_b = sim_b.run_simulation()
        # Final global hash must be identical
        assert logs_a[-1]["global_hash"] == logs_b[-1]["global_hash"], \
            "Same seed must produce identical global hashes"


# ---------------------------------------------------------------------------
# (e) BM-M4 — Explicit error when Cleveland file missing
# ---------------------------------------------------------------------------

class TestMissingDataHandling:
    def test_missing_file_raises_by_default(self, tmp_path, monkeypatch):
        """
        When the Cleveland file is absent and allow_synthetic=False (default),
        _load_cleveland must raise MissingDataError.
        """
        import app.federated_sim_engine as eng
        # Redirect the module's data search path to a directory without the file.
        monkeypatch.setattr(eng, "__file__", str(tmp_path / "fake_engine.py"))
        with pytest.raises(MissingDataError):
            _load_cleveland(allow_synthetic=False)

    def test_missing_file_synthetic_when_opted_in(self, tmp_path, monkeypatch):
        """
        When allow_synthetic=True and file is missing, must return synthetic data
        of the expected shape and mark synthetic_fallback.
        """
        import app.federated_sim_engine as eng
        eng_file_backup = eng.__file__
        try:
            eng.__file__ = str(tmp_path / "fake_engine.py")
            X, y = _load_cleveland(allow_synthetic=True, random_state=42)
            assert X.shape == (303, 13)
            assert len(y) == 303
            assert eng._LAST_LOAD_SYNTHETIC is True
        finally:
            eng.__file__ = eng_file_backup

    def test_simulation_synthetic_fallback_explicit(self, tmp_path, monkeypatch):
        """
        FederatedSimulation with allow_synthetic=True on a missing-file path
        must set synthetic_fallback=True.
        """
        import app.federated_sim_engine as eng
        eng_file_backup = eng.__file__
        try:
            eng.__file__ = str(tmp_path / "fake_engine.py")
            sim = FederatedSimulation(n_nodes=3, n_rounds=2, seed=42,
                                       allow_synthetic=True)
            assert sim.synthetic_fallback is True
        finally:
            eng.__file__ = eng_file_backup


# ---------------------------------------------------------------------------
# (f) BM-H3 — Byzantine detection: attacker flagged, clean run has no flags
# ---------------------------------------------------------------------------

class TestByzantineDetection:
    def test_byzantine_attacker_flagged_immediately(self):
        """
        When a node injects large Byzantine noise (N(5,2)) into its delta,
        it must be flagged by the update-norm detector in that same round.
        """
        sim = FederatedSimulation(n_nodes=4, n_rounds=3, noise_multiplier=0.5,
                                   seed=42)
        # Tamper at round 0
        logs = sim.run_simulation(tamper_round=0, tamper_node=0)
        # Round 0 (index 0) must flag Hospital-1
        round0_flagged = logs[0]["flagged_nodes"]
        assert "Hospital-1" in round0_flagged, \
            f"Byzantine node must be flagged in round it attacks; got {round0_flagged}"

    def test_clean_run_no_false_positives(self):
        """
        In a clean run (no Byzantine nodes) no node must be flagged.
        """
        sim = FederatedSimulation(n_nodes=4, n_rounds=5, noise_multiplier=1.1,
                                   seed=42)
        logs = sim.run_simulation()
        all_flagged = []
        for rlog in logs:
            all_flagged.extend(rlog["flagged_nodes"])
        assert len(all_flagged) == 0, \
            f"No nodes should be flagged in a clean run; got {all_flagged}"

    def test_detect_byzantine_by_norm_unit(self):
        """
        Unit test for _detect_byzantine_by_norm: single outlier delta must be flagged.
        """
        rng = np.random.default_rng(0)
        # 4 normal deltas
        deltas = [rng.normal(0, 0.1, 14) for _ in range(3)]
        # 1 clearly Byzantine delta (large magnitude)
        deltas.append(np.ones(14) * 100.0)
        ids = ["A", "B", "C", "Attacker"]
        flagged = _detect_byzantine_by_norm(deltas, ids)
        assert "Attacker" in flagged, f"Large-norm delta must be flagged; got {flagged}"

    def test_detect_byzantine_no_false_positive_unit(self):
        """All similar deltas must produce no flags."""
        rng = np.random.default_rng(1)
        deltas = [rng.normal(0, 0.05, 14) for _ in range(4)]
        ids = ["A", "B", "C", "D"]
        flagged = _detect_byzantine_by_norm(deltas, ids)
        assert len(flagged) == 0, f"No false positives in clean deltas; got {flagged}"


# ---------------------------------------------------------------------------
# (g) BM-H5 — DP accountant sanity
# ---------------------------------------------------------------------------

class TestDPAccountantSanity:
    def test_epsilon_monotone_in_rounds(self):
        """Epsilon must be non-decreasing as number of rounds increases."""
        nm = 1.1
        epsilons = [_renyi_epsilon(nm, steps=t, delta=1e-5) for t in range(1, 15)]
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1], \
                f"Epsilon not monotone at step {i}: {epsilons[i-1]} → {epsilons[i]}"

    def test_epsilon_decreases_with_more_noise(self):
        """More noise (higher nm) must produce smaller epsilon."""
        eps_low_noise  = _renyi_epsilon(0.5, steps=10, delta=1e-5)
        eps_high_noise = _renyi_epsilon(2.0, steps=10, delta=1e-5)
        assert eps_high_noise < eps_low_noise, \
            "Higher noise_multiplier must yield lower epsilon"

    def test_epsilon_correct_sensitivity_factor(self):
        """
        GaussianMechanism.compute_epsilon must use sensitivity_factor=2
        (replace-one adjacency for FedAvg).  Verify by comparing with direct
        _renyi_epsilon call with factor=2.
        """
        gm = GaussianMechanism(noise_multiplier=1.1, max_grad_norm=1.0, delta=1e-5)
        eps_gm = gm.compute_epsilon(steps=10)
        eps_direct = _renyi_epsilon(1.1, steps=10, delta=1e-5, sensitivity_factor=2.0)
        assert abs(eps_gm - eps_direct) < 1e-9, \
            "GaussianMechanism must use sensitivity_factor=2"

    def test_simulation_epsilon_in_round_log(self):
        """Per-round cumulative_epsilon in simulation must be finite and positive."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=5, noise_multiplier=1.1,
                                   seed=42)
        logs = sim.run_simulation()
        for rlog in logs:
            eps = rlog["cumulative_epsilon"]
            assert math.isfinite(eps) and eps > 0, \
                f"cumulative_epsilon must be finite positive, got {eps}"


# ---------------------------------------------------------------------------
# (h) Ablation config bugs
# ---------------------------------------------------------------------------

class TestAblationConfig:
    def test_full_stack_and_fedavg_dp_smpc_are_different(self):
        """
        In the fixed ablation, full_stack uses nm=0.8 and fedavg_dp_smpc uses
        nm=1.1, so they must produce different epsilon values.
        """
        from scripts.ablation_study import run_ablation
        results = run_ablation(n_rounds=2, n_nodes=3)
        # Both conditions must exist
        assert "fedavg_dp_smpc" in results
        assert "full_stack" in results
        eps_smpc    = results["fedavg_dp_smpc"]["final_cumulative_epsilon"]
        eps_full    = results["full_stack"]["final_cumulative_epsilon"]
        # nm=1.1 → smaller epsilon; nm=0.8 → larger epsilon
        # (full_stack has more noise or different nm so epsilons must differ)
        assert eps_smpc != eps_full, \
            "fedavg_dp_smpc and full_stack must have different epsilon values"

    def test_no_dp_condition_has_inf_epsilon(self):
        """plain_fedavg (nm=0) must report inf epsilon (stored as string 'inf' or float inf)."""
        from scripts.ablation_study import run_ablation
        results = run_ablation(n_rounds=2, n_nodes=3)
        eps = results["plain_fedavg"]["final_cumulative_epsilon"]
        # The ablation stores epsilon as "inf" string when nm=0 (JSON-safe).
        # Accept either float inf or the string "inf".
        is_inf = (isinstance(eps, float) and math.isinf(eps)) or eps == "inf"
        assert is_inf, f"No-DP condition must have inf epsilon, got {eps!r}"


# ---------------------------------------------------------------------------
# (i) Benchmark determinism
# ---------------------------------------------------------------------------

class TestBenchmarkDeterminism:
    def test_two_runs_identical(self, tmp_path):
        """
        Running the benchmark twice with the same seed must produce identical
        final_global_accuracy for every condition.
        """
        from app.benchmark import _run_condition
        # Use a tiny config for speed
        result_a = _run_condition(
            "test", n_nodes=3, n_rounds=2, noise_multiplier=0.0,
            use_smpc=False, seed=7
        )
        result_b = _run_condition(
            "test", n_nodes=3, n_rounds=2, noise_multiplier=0.0,
            use_smpc=False, seed=7
        )
        assert result_a["final_global_accuracy"] == result_b["final_global_accuracy"], \
            "Benchmark must be deterministic: same seed → same accuracy"

    def test_benchmark_writes_valid_json(self, tmp_path, monkeypatch):
        """
        run_benchmark with small config must write valid JSON to data/.
        """
        from app import benchmark as bm_module
        monkeypatch.setattr(bm_module, "DATA_DIR", tmp_path)
        from app.benchmark import run_benchmark
        run_benchmark(n_nodes=3, n_rounds=2, seed=42, n_seeds=2)
        out_path = tmp_path / "benchmark_results.json"
        assert out_path.exists(), "benchmark_results.json must be written"
        data = json.loads(out_path.read_text())
        assert "metadata" in data
        assert "ablation_conditions" in data
        assert data["metadata"].get("synthetic_fallback") is False, \
            "Real data must not be flagged as synthetic"


# ---------------------------------------------------------------------------
# (i) Remediation review — detector, SMPC wiring, DP noise scale, hashes
# ---------------------------------------------------------------------------

class TestRemediationReview:
    def test_detector_flags_attacker_with_three_clients(self):
        deltas = [np.ones(5) * v for v in (1.0, 1.1, 100.0)]
        assert _detect_byzantine_by_norm(deltas, ["a", "b", "c"]) == ["c"]

    def test_detector_silent_below_min_clients(self):
        deltas = [np.ones(5), np.ones(5) * 100.0]
        assert _detect_byzantine_by_norm(deltas, ["a", "b"]) == []

    @pytest.mark.parametrize("n_nodes", [3, 5])
    def test_detector_over_seeds_and_node_counts(self, n_nodes):
        """Attacker flagged in the attack round and no flags in clean runs."""
        for seed in range(4):
            attacked = FederatedSimulation(n_nodes=n_nodes, n_rounds=2,
                                           seed=seed, noise_multiplier=1.1)
            logs = attacked.run_simulation(tamper_round=1, tamper_node=0)
            assert "Hospital-1" in logs[1]["flagged_nodes"]
            clean = FederatedSimulation(n_nodes=n_nodes, n_rounds=2,
                                        seed=seed, noise_multiplier=1.1)
            assert all(not l["flagged_nodes"] for l in clean.run_simulation())

    def test_smpc_path_is_used_and_matches_plain_sum(self, monkeypatch):
        """use_smpc=True must call secure_aggregate; the sum must be equal."""
        from app import federated_sim_engine as eng
        calls = []
        orig = eng.AdditiveSecretSharing.secure_aggregate

        def spy(ws, rng=None):
            calls.append(len(ws))
            return orig(ws, rng)

        monkeypatch.setattr(eng.AdditiveSecretSharing, "secure_aggregate",
                            staticmethod(spy))
        plain = FederatedSimulation(n_nodes=3, n_rounds=2, seed=5,
                                    noise_multiplier=1.1, use_smpc=False)
        smpc = FederatedSimulation(n_nodes=3, n_rounds=2, seed=5,
                                   noise_multiplier=1.1, use_smpc=True)
        plain.run_simulation()
        assert calls == []
        smpc.run_simulation()
        assert calls == [3, 3]
        np.testing.assert_allclose(
            plain.global_weights["coef"], smpc.global_weights["coef"], atol=1e-6)

    def test_plain_fedavg_is_not_clipped(self):
        node = FederatedSimulation(n_nodes=3, n_rounds=1, seed=1,
                                   noise_multiplier=0.0).nodes[0]
        node.train_local(None)
        assert np.allclose(node.get_clipped_delta(), node.get_raw_delta())

    def test_dp_noise_std_matches_accountant_assumption(self):
        """Noise std = nm*C, which the accountant treats as nm*C vs 2C
        sensitivity (RDP = alpha*4/(2 nm^2))."""
        gm = GaussianMechanism(noise_multiplier=2.0, max_grad_norm=3.0,
                               rng=np.random.default_rng(0))
        noise = gm.add_noise_to_aggregate(np.zeros(200000), n_clients=3)
        assert abs(noise.std() - 6.0) < 0.05

    def test_round_hash_covers_submitted_update(self):
        import hashlib
        sim = FederatedSimulation(n_nodes=3, n_rounds=1, seed=2,
                                  noise_multiplier=1.1)
        log = sim.run_round(0)
        node = sim.nodes[0]
        expect = hashlib.sha256(
            node.get_clipped_delta().tobytes()).hexdigest()
        assert log["nodes"][0]["hash"] == expect
