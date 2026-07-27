"""
Unit tests for differential-privacy mechanics inside federated_sim_engine.

All tests are self-contained (no external services required).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import numpy as np
import pytest

from app.federated_sim_engine import GaussianMechanism, AdditiveSecretSharing, FederatedNode


# ---------------------------------------------------------------------------
# GaussianMechanism
# ---------------------------------------------------------------------------

class TestGaussianMechanism:
    def test_noise_is_nonzero(self):
        """Adding Gaussian noise must change the weights."""
        gm = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
        w = np.ones(50)
        noisy = gm.clip_and_noise(w.copy())
        assert not np.allclose(w, noisy), "Noise added must alter weights"

    def test_noise_scale_increases_with_higher_multiplier(self):
        """Higher noise_multiplier → more noise → higher mean absolute deviation."""
        rng = np.random.default_rng(7)
        base = rng.standard_normal(500)

        gm_low  = GaussianMechanism(noise_multiplier=0.1, max_grad_norm=10.0, delta=1e-5)
        gm_high = GaussianMechanism(noise_multiplier=3.0, max_grad_norm=10.0, delta=1e-5)

        noise_low  = np.mean(np.abs(gm_low.clip_and_noise(base.copy())  - base))
        noise_high = np.mean(np.abs(gm_high.clip_and_noise(base.copy()) - base))
        assert noise_high > noise_low, "Higher noise_multiplier should inject more noise"

    def test_epsilon_accounting_positive(self):
        """compute_epsilon must return a positive finite value."""
        gm = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
        eps = gm.compute_epsilon(steps=5)
        assert eps > 0, "Epsilon must be positive"
        assert math.isfinite(eps), "Epsilon must be finite"

    def test_epsilon_grows_with_rounds(self):
        """More FL rounds accumulate more privacy budget."""
        gm = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
        eps_5  = gm.compute_epsilon(steps=5)
        eps_20 = gm.compute_epsilon(steps=20)
        assert eps_20 > eps_5, "Cumulative epsilon must grow with more rounds"

    def test_clipping_limits_l2_norm(self):
        """After gradient clipping, L2 norm must be ≤ max_grad_norm."""
        clip_norm = 1.0
        # Use noise_multiplier=0 so no randomness is added — we just test clipping
        gm = GaussianMechanism(noise_multiplier=0.0, max_grad_norm=clip_norm, delta=1e-5)
        large = np.full(100, 10.0)
        clipped = gm.clip_and_noise(large.copy())
        assert np.linalg.norm(clipped) <= clip_norm + 1e-6, \
            "Clipped norm must not exceed max_grad_norm"

    def test_no_clip_when_below_threshold(self):
        """Weights already within clip norm should not be scaled down."""
        gm = GaussianMechanism(noise_multiplier=0.0, max_grad_norm=100.0, delta=1e-5)
        small = np.array([0.1, 0.2, 0.3])
        result = gm.clip_and_noise(small.copy())
        # With zero noise, should be equal (no clipping)
        np.testing.assert_allclose(result, small, rtol=1e-6)


# ---------------------------------------------------------------------------
# AdditiveSecretSharing
# ---------------------------------------------------------------------------

class TestAdditiveSecretSharing:
    def test_reconstruction_exact(self):
        """Sum of all shares must reconstruct the original secret exactly."""
        secret = np.array([1.5, -2.3, 0.0, 100.0])
        shares = AdditiveSecretSharing.share(secret, n_parties=4)
        reconstructed = AdditiveSecretSharing.reconstruct(shares)
        np.testing.assert_allclose(reconstructed, secret, atol=1e-9,
                                   err_msg="SMPC reconstruction must recover original weights")

    def test_shares_differ_from_secret(self):
        """Individual shares (except possibly the last) should not equal the secret."""
        secret = np.ones(20) * 3.14
        shares = AdditiveSecretSharing.share(secret, n_parties=3)
        # At least two of the three shares should differ from the secret
        non_matching = sum(1 for s in shares if not np.allclose(s, secret))
        assert non_matching >= 2, "Most shares should not reveal the secret directly"

    def test_correct_number_of_shares(self):
        """share() must produce exactly n_parties shares."""
        shares = AdditiveSecretSharing.share(np.zeros(10), n_parties=5)
        assert len(shares) == 5

    def test_reconstruction_with_two_parties(self):
        secret = np.linspace(-1, 1, 50)
        shares = AdditiveSecretSharing.share(secret, n_parties=2)
        reconstructed = AdditiveSecretSharing.reconstruct(shares)
        np.testing.assert_allclose(reconstructed, secret, atol=1e-9)

    def test_secure_aggregate_matches_plain_mean(self):
        """secure_aggregate must equal the plain mean of inputs."""
        rng = np.random.default_rng(42)
        weights = [rng.standard_normal(20) for _ in range(4)]
        plain_avg = np.mean(weights, axis=0)
        smpc_avg = AdditiveSecretSharing.secure_aggregate(weights)
        np.testing.assert_allclose(smpc_avg, plain_avg, atol=1e-7,
                                   err_msg="SMPC aggregate must match plain FedAvg")


# ---------------------------------------------------------------------------
# FederatedNode (unit — no FL simulation, just local training)
# ---------------------------------------------------------------------------

class TestFederatedNode:
    def test_node_trains_and_returns_weights(self, small_dataset):
        """A single node must have weights after local training."""
        X, y = small_dataset
        node = FederatedNode(node_id="TestHospital", X=X, y=y)
        node.train_local(global_weights=None)
        assert node.weights is not None, "train_local must populate weights"
        assert len(node.weights.flatten()) > 0, "Weight vector must not be empty"

    def test_node_accuracy_is_valid(self, small_dataset):
        """Node evaluation accuracy must be between 0 and 1."""
        X, y = small_dataset
        node = FederatedNode(node_id="TestHospital-2", X=X, y=y)
        node.train_local(global_weights=None)
        metrics = node.evaluate()
        assert 0.0 <= metrics["accuracy"] <= 1.0, \
            f"Accuracy out of range: {metrics['accuracy']}"

    def test_dp_weights_differ_from_raw(self, small_dataset):
        """DP-noised weights must differ from the raw model weights."""
        X, y = small_dataset
        gm = GaussianMechanism(noise_multiplier=1.0, max_grad_norm=1.0, delta=1e-5)
        node = FederatedNode(node_id="TestHospital-3", X=X, y=y, dp=gm)
        node.train_local(global_weights=None)
        raw_flat = np.concatenate([node.weights.flatten(), node.intercept.flatten()])
        dp_flat = node.get_dp_weights()
        assert not np.allclose(raw_flat, dp_flat), \
            "DP weights must differ from raw weights"

    def test_model_hash_is_hex_string(self, small_dataset):
        """get_model_hash must return a 64-char hex SHA-256 string."""
        X, y = small_dataset
        node = FederatedNode(node_id="TestHospital-4", X=X, y=y)
        node.train_local()
        node.get_dp_weights()
        h = node.get_model_hash()
        assert isinstance(h, str), "Hash must be a string"
        assert len(h) == 64, f"SHA-256 hex must be 64 chars, got {len(h)}"
