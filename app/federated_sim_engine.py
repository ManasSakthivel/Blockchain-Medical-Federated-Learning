# federated_sim_engine.py
# Safe, modular simulation engine for multi-round federated learning
# Does NOT affect your main app logic

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import hashlib
import random
import copy

class FederatedNode:
    def __init__(self, node_id, n_samples=100, n_features=5, random_state=None):
        self.node_id = node_id
        self.X, self.y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=3, n_redundant=0, random_state=random_state)
        self.model = LogisticRegression(max_iter=100)
        self.weights = None
        self.intercept = None
        self.hashes = []
        self.accuracies = []
        self.status = 'Initialized'
        self.tampered = False

    def train_local(self, global_weights=None):
        if global_weights is not None:
            self.model.coef_ = global_weights['coef']
            self.model.intercept_ = global_weights['intercept']
        self.model.fit(self.X, self.y)
        self.weights = copy.deepcopy(self.model.coef_)
        self.intercept = copy.deepcopy(self.model.intercept_)
        self.status = 'Trained'

    def evaluate(self):
        preds = self.model.predict(self.X)
        acc = accuracy_score(self.y, preds)
        self.accuracies.append(acc)
        return acc

    def get_model_hash(self):
        w_bytes = np.concatenate([self.weights.flatten(), self.intercept.flatten()]).tobytes()
        h = hashlib.sha256(w_bytes).hexdigest()
        if self.tampered:
            # Simulate tampering by flipping a bit
            h = h[:-1] + random.choice('0123456789abcdef')
        self.hashes.append(h)
        return h

    def tamper(self):
        self.tampered = True
        self.status = 'Tampered'

class FederatedSimulation:
    def __init__(self, n_nodes=3, n_rounds=3, n_features=5):
        self.nodes = [FederatedNode(f'Hospital {i+1}', n_features=n_features, random_state=i) for i in range(n_nodes)]
        self.n_rounds = n_rounds
        self.global_weights = None
        self.global_hashes = []
        self.round_logs = []

    def run_round(self, round_idx, tamper_node=None):
        local_weights = []
        local_hashes = []
        round_log = {'round': round_idx+1, 'nodes': []}
        for i, node in enumerate(self.nodes):
            if tamper_node is not None and i == tamper_node:
                node.tamper()
            node.train_local(self.global_weights)
            acc = node.evaluate()
            h = node.get_model_hash()
            local_weights.append(np.concatenate([node.weights.flatten(), node.intercept.flatten()]))
            local_hashes.append(h)
            round_log['nodes'].append({
                'id': node.node_id,
                'hash': h,
                'accuracy': acc,
                'status': node.status
            })
        # Aggregate global weights
        agg_weights = np.mean(local_weights, axis=0)
        n_coef = self.nodes[0].weights.shape[1]
        global_coef = agg_weights[:n_coef].reshape(1, -1)
        global_intercept = agg_weights[n_coef:].reshape(1,)
        self.global_weights = {'coef': global_coef, 'intercept': global_intercept}
        # Hash global model
        w_bytes = agg_weights.tobytes()
        global_hash = hashlib.sha256(w_bytes).hexdigest()
        self.global_hashes.append(global_hash)
        round_log['global_hash'] = global_hash
        self.round_logs.append(round_log)
        return round_log

    def run_simulation(self, tamper_round=None, tamper_node=None):
        self.global_weights = None
        self.global_hashes = []
        self.round_logs = []
        for r in range(self.n_rounds):
            if tamper_round is not None and r == tamper_round:
                log = self.run_round(r, tamper_node=tamper_node)
            else:
                log = self.run_round(r)
        return self.round_logs
