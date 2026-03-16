"""
benchmark.py
Benchmarking script for the Federated Learning simulation engine.
Runs FL simulation across multiple rounds and records accuracy,
communication overhead, and a simulated privacy loss metric.
"""

import json
import numpy as np
from app.federated_sim_engine import FederatedSimulation


def run_benchmark(n_nodes=3, n_rounds=10, n_features=5):
    """
    Run the federated learning simulation benchmark.
    Returns a dict with accuracy, communication, and privacy_loss per round.
    """
    sim = FederatedSimulation(n_nodes=n_nodes, n_rounds=n_rounds, n_features=n_features)
    round_logs = sim.run_simulation()

    accuracy_per_round = []
    communication_per_round = []
    privacy_loss_per_round = []

    for round_log in round_logs:
        # Average node accuracy for this round
        node_accuracies = [node['accuracy'] for node in round_log['nodes']]
        avg_accuracy = float(np.mean(node_accuracies))
        accuracy_per_round.append(round(avg_accuracy, 4))

        # Simulated communication overhead (bytes) — proportional to the number of
        # model parameters exchanged. Using n_features as a proxy.
        comm_overhead = int(n_nodes * n_features * 8)  # 8 bytes per float64
        communication_per_round.append(comm_overhead)

        # Simulated privacy loss (epsilon) — decreases with more rounds (more noise averaging)
        # This is illustrative. In a real DP-FL setup you'd track actual epsilon.
        epsilon = round(max(0.01, 0.1 / (round_log['round'] ** 0.5)), 4)
        privacy_loss_per_round.append(epsilon)

    results = {
        "n_nodes": n_nodes,
        "n_rounds": n_rounds,
        "accuracy": accuracy_per_round,
        "communication": communication_per_round,
        "privacy_loss": privacy_loss_per_round,
        "final_global_hash": sim.global_hashes[-1] if sim.global_hashes else None,
    }

    # Save results to file
    output_path = "app/benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Benchmark complete. Results saved to {output_path}")
    print(f"   Final avg accuracy: {accuracy_per_round[-1]:.4f}")
    print(f"   Final privacy loss (epsilon): {privacy_loss_per_round[-1]:.4f}")
    print(f"   Global model hash (last round): {results['final_global_hash']}")

    return results


if __name__ == "__main__":
    run_benchmark(n_nodes=3, n_rounds=10, n_features=13)