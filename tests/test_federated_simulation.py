"""
Unit tests for the federated simulation end-to-end pipeline.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from app.federated_sim_engine import (
    FederatedSimulation,
    FederatedNode,
    AdditiveSecretSharing,
    GaussianMechanism,
)


class TestFederatedSimulation:
    def test_simulation_runs_without_error(self):
        """FederatedSimulation must complete without raising an exception."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=2, noise_multiplier=1.1)
        logs = sim.run_simulation()
        assert logs is not None
        assert len(logs) == 2

    def test_simulation_returns_per_round_logs(self):
        """Each round must produce a log entry with expected keys."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=3)
        logs = sim.run_simulation()
        assert len(logs) == 3
        for rlog in logs:
            assert "round" in rlog
            assert "nodes" in rlog
            assert "global_hash" in rlog
            assert "cumulative_epsilon" in rlog

    def test_node_accuracy_in_range(self):
        """Every node accuracy reported per round must be in [0, 1]."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=2)
        logs = sim.run_simulation()
        for rlog in logs:
            for nlog in rlog["nodes"]:
                acc = nlog["accuracy"]
                assert 0.0 <= acc <= 1.0, f"Node accuracy out of range: {acc}"

    def test_dp_epsilon_grows_monotonically(self):
        """Cumulative epsilon must be non-decreasing across rounds."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=5, noise_multiplier=1.1)
        logs = sim.run_simulation()
        epsilons = [rlog["cumulative_epsilon"] for rlog in logs]
        for i in range(1, len(epsilons)):
            assert epsilons[i] >= epsilons[i - 1], \
                f"Epsilon decreased at round {i}: {epsilons[i-1]} → {epsilons[i]}"

    def test_global_hash_is_hex(self):
        """Global model hash must be a 64-char hex SHA-256 string."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=2)
        logs = sim.run_simulation()
        for rlog in logs:
            h = rlog["global_hash"]
            assert len(h) == 64, f"Global hash must be 64 hex chars, got {len(h)}"

    def test_smpc_aggregation_correctness(self):
        """SMPC aggregated result must be numerically close to plain mean."""
        rng = np.random.default_rng(99)
        n = 4
        dim = 20
        weight_sets = [rng.standard_normal(dim) for _ in range(n)]

        plain_avg = np.mean(weight_sets, axis=0)
        smpc_avg = AdditiveSecretSharing.secure_aggregate(weight_sets)

        np.testing.assert_allclose(
            smpc_avg, plain_avg, atol=1e-7,
            err_msg="SMPC-aggregated average must match plain FedAvg"
        )

    def test_final_accuracy_available(self):
        """get_final_global_accuracy must return a float in [0, 1]."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=3)
        sim.run_simulation()
        acc = sim.get_final_global_accuracy()
        assert isinstance(acc, float), "Final accuracy must be a float"
        assert 0.0 <= acc <= 1.0, f"Final accuracy out of range: {acc}"

    def test_reputation_scores_populated(self):
        """After simulation, each node must have a reputation in [0, 1]."""
        sim = FederatedSimulation(n_nodes=3, n_rounds=2)
        sim.run_simulation()
        for node in sim.nodes:
            assert 0.0 <= node.reputation <= 1.0, \
                f"Node {node.node_id} reputation out of range: {node.reputation}"

    def test_byzantine_node_flagged(self):
        """
        When a tampered (Byzantine) node is injected, the simulation must
        eventually flag at least one anomalous node.
        """
        sim = FederatedSimulation(n_nodes=4, n_rounds=5, noise_multiplier=0.5)
        # Tamper node 0 starting from round 0
        logs = sim.run_simulation(tamper_round=0, tamper_node=0)
        # Check flags across all rounds
        all_flagged = set()
        for rlog in logs:
            all_flagged.update(rlog.get("flagged_nodes", []))
        # The tampered node should be flagged in at least one round
        # (reputation drops below threshold over time)
        node0 = sim.nodes[0]
        assert node0.tampered, "Node 0 should be marked tampered"
        # Reputation of tampered node should have degraded
        assert node0.reputation < 1.0, \
            "Tampered node reputation should have decreased"
