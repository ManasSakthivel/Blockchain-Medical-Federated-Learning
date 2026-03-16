"""
federated_simulation.py
Federated Learning Simulation for EHR Blockchain Project

Simulates multiple hospitals training local models, hashing their model weights,
and recording those hashes on the FederatedLearning smart contract via Ganache.

This replaces the old script where blockchain calls were commented out.
The FederatedLearning.sol contract must be deployed before running this.
"""

import os
import sys
import hashlib
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from web3 import Web3


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

GANACHE_URL = os.environ.get("GANACHE_URL", "http://127.0.0.1:7545")
BUILD_PATH  = os.path.join(os.path.dirname(__file__), '..', 'build', 'contracts')


# ──────────────────────────────────────────────────────────────────────────────
# Blockchain helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_private_key(web3: Web3, account: str) -> str | None:
    """
    Load the private key for `account` from the environment variable first,
    then fall back to the private_keys.json file (dev-only, never commit that file).
    Returns None if not found.
    """
    # Preferred: read from env  e.g. GANACHE_PRIVATE_KEY_0xABCD...=0xDEAD...
    env_key = f"GANACHE_PRIVATE_KEY_{account}"
    pk = os.environ.get(env_key)
    if pk:
        return pk

    # Dev fallback: read from private_keys.json (should be git-ignored)
    keys_path = os.path.join(os.path.dirname(__file__), '..', 'private_keys.json')
    if os.path.exists(keys_path):
        with open(keys_path, 'r') as f:
            keys = json.load(f)
        return keys.get(account)

    return None


def _load_federated_contract(web3: Web3):
    """
    Load the deployed FederatedLearning contract from Truffle build artifacts.
    Returns (contract, address) or raises RuntimeError.
    """
    contract_json_path = os.path.join(BUILD_PATH, 'FederatedLearning.json')
    if not os.path.exists(contract_json_path):
        raise RuntimeError(
            f"❌ FederatedLearning.json not found at {contract_json_path}.\n"
            "   Run: truffle compile && truffle migrate --network development"
        )

    with open(contract_json_path, 'r') as f:
        contract_data = json.load(f)

    networks = contract_data.get('networks', {})
    if not networks:
        raise RuntimeError(
            "❌ FederatedLearning contract has not been deployed.\n"
            "   Run: truffle migrate --network development"
        )

    latest_network = max(networks.keys(), key=int)
    address = networks[latest_network]['address']
    abi     = contract_data['abi']

    contract = web3.eth.contract(address=address, abi=abi)
    print(f"✅ Loaded FederatedLearning contract at {address}")
    return contract, address


# ──────────────────────────────────────────────────────────────────────────────
# Federated Learning logic
# ──────────────────────────────────────────────────────────────────────────────

def generate_hospital_data(n_samples: int = 100, n_features: int = 13,
                           random_state: int | None = None):
    """Simulate a hospital's local dataset (Cleveland Heart Disease feature space)."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        random_state=random_state
    )
    return X, y


def train_local_model(X, y):
    """Train a logistic regression model on local data; return flattened weights."""
    model = LogisticRegression(max_iter=500)
    model.fit(X, y)
    return model.coef_.flatten(), model.intercept_.flatten()


def hash_model(weights) -> str:
    """SHA-256 hash of the model weight vector (as raw bytes)."""
    return hashlib.sha256(np.array(weights, dtype=np.float64).tobytes()).hexdigest()


def record_hash_on_chain(web3: Web3, contract, account: str,
                         model_hash_hex: str, round_num: int) -> str | None:
    """
    Call FederatedLearning.recordModelUpdate(bytes32 modelHash, uint round).
    Returns the transaction hash string, or None on failure.
    """
    private_key = _load_private_key(web3, account)
    if not private_key:
        print(f"⚠️  No private key found for {account}. Skipping on-chain recording.")
        return None

    try:
        # Convert hex digest to bytes32
        model_hash_bytes = bytes.fromhex(model_hash_hex)

        tx = contract.functions.recordModelUpdate(
            model_hash_bytes, round_num
        ).build_transaction({
            'from': account,
            'gas': 200_000,
            'gasPrice': web3.eth.gas_price,
            'nonce': web3.eth.get_transaction_count(account),
        })

        signed = web3.eth.account.sign_transaction(tx, private_key=private_key)
        tx_hash = web3.eth.send_raw_transaction(signed.raw_transaction)
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        tx_hex = receipt.transactionHash.hex()
        print(f"   ⛓️  On-chain ✅  tx={tx_hex[:18]}…")
        return tx_hex

    except Exception as e:
        print(f"   ⛓️  On-chain ❌  {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Main simulation
# ──────────────────────────────────────────────────────────────────────────────

def federated_round(hospitals, round_num: int, web3: Web3 | None,
                    contract, account: str | None) -> dict:
    """Run one federated learning round across all hospitals."""
    local_hashes  = []
    local_weights = []

    print(f"\n━━━ Round {round_num} ━━━")
    for i, (X, y) in enumerate(hospitals):
        weights, intercept = train_local_model(X, y)
        all_weights = np.concatenate([weights, intercept])
        model_hash  = hash_model(all_weights)

        print(f"  🏥 Hospital {i + 1}  accuracy≈{_eval_acc(X, y, weights, intercept):.3f}"
              f"  hash={model_hash[:16]}…")

        # Record each hospital's local hash on-chain
        if web3 and contract and account:
            record_hash_on_chain(web3, contract, account, model_hash, round_num)

        local_hashes.append(model_hash)
        local_weights.append(all_weights)

    # FedAvg: aggregate weights
    global_weights = np.mean(local_weights, axis=0)
    global_hash    = hash_model(global_weights)
    print(f"  🌐 Global model   hash={global_hash[:16]}…")

    # Record global aggregated hash on-chain
    if web3 and contract and account:
        record_hash_on_chain(web3, contract, account, global_hash, round_num)

    return {
        'round':         round_num,
        'local_hashes':  local_hashes,
        'global_hash':   global_hash,
        'global_weights': global_weights.tolist(),
    }


def _eval_acc(X, y, coef, intercept) -> float:
    """Quick training accuracy for logging."""
    preds = (X @ coef + intercept[0]) > 0
    return float((preds == y).mean())


def run_simulation(n_hospitals: int = 3, n_rounds: int = 3,
                   n_features: int = 13, enable_blockchain: bool = True) -> list[dict]:
    """
    Full federated learning simulation.

    Parameters
    ----------
    n_hospitals       number of hospital nodes
    n_rounds          number of FL rounds
    n_features        feature dimensions (13 = Cleveland Heart Disease space)
    enable_blockchain whether to record hashes on FederatedLearning.sol
    """
    # Generate synthetic hospital datasets
    hospitals = [generate_hospital_data(n_features=n_features, random_state=i)
                 for i in range(n_hospitals)]

    web3 = contract = account = None

    if enable_blockchain:
        web3 = Web3(Web3.HTTPProvider(GANACHE_URL))
        if not web3.is_connected():
            print(f"⚠️  Ganache not reachable at {GANACHE_URL}. "
                  "Continuing WITHOUT blockchain recording.")
            web3 = None
        else:
            print(f"✅ Connected to Ganache at {GANACHE_URL}")
            try:
                contract, _ = _load_federated_contract(web3)
                accounts = web3.eth.accounts
                account  = accounts[0] if accounts else None
                print(f"✅ Using account: {account}")
            except RuntimeError as e:
                print(str(e))
                print("   Continuing WITHOUT blockchain recording.")
                contract = account = None

    round_logs = []
    for r in range(1, n_rounds + 1):
        log = federated_round(hospitals, r, web3, contract, account)
        round_logs.append(log)

    print("\n✅ Simulation complete.")
    if contract:
        total = contract.functions.getUpdateCount().call()
        print(f"   FederatedLearning contract total on-chain updates: {total}")

    return round_logs


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logs = run_simulation(n_hospitals=3, n_rounds=3, n_features=13,
                          enable_blockchain=True)
    print("\nRound summary:")
    for log in logs:
        print(f"  Round {log['round']}: global_hash={log['global_hash'][:24]}…")
