# Blockchain-Powered Decentralized Federated Learning for Medical AI Systems

[![CI](https://github.com/ManasSakthivel/Blockchain-Medical-Federated-Learning/actions/workflows/ci.yml/badge.svg)](https://github.com/ManasSakthivel/Blockchain-Medical-Federated-Learning/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A production-grade implementation of the research paper:

> **"Blockchain-Powered Decentralized Federated Learning for Medical AI Systems"**
> — accepted Q3 journal, 2024.

The system combines **differential privacy (DP)**, **secure multi-party computation
(SMPC)**, **smart-contract-enforced reputation**, and **GDPR-compliant consent
management** to protect patient data while training high-accuracy ML models across
hospital nodes — without any raw data ever leaving its origin silo.

---

## Features

| Layer | Technology | What it provides |
|---|---|---|
| **FL Engine** | FedAvg + Gaussian Mechanism | Real privacy budget accounting (ε, δ) |
| **SMPC** | Additive Secret Sharing | Aggregator never sees raw gradients |
| **Reputation** | EMA per round | Detects & flags Byzantine nodes |
| **Smart Contracts** | Solidity `^0.8.20` + Truffle | Immutable audit trail + slashing |
| **Consent** | `ConsentContract.sol` | GDPR Art. 7 — grant / revoke per recipient |
| **GDPR Erasure** | Functional erasure model | Art. 17 — "Right to be Forgotten" |
| **DIDs** | `did:ethr:<address>` | W3C Decentralized Identifiers at registration |
| **Dataset** | UCI Cleveland Heart Disease | 303 patients, 13 features, binary target |
| **Storage** | IPFS | Decentralised content-addressed file storage |
| **Blockchain** | Ganache (dev) / Sepolia (testnet) | Ethereum-compatible |

---

## Research Paper

The implementation is grounded in the following claims from the paper:

- **zk-STARK equivalent** — SHA-256 model-hash anchoring on `FederatedLearning.sol`
- **Differential Privacy** — Gaussian mechanism with Rényi advanced composition
- **SMPC aggregation** — (n,n)-threshold additive secret sharing
- **Reputation consensus** — EMA-based scoring, slashing on detection
- **Layer-2 scalability** — Sepolia testnet configuration in `truffle-config.js`
- **MIMIC-III / Cleveland evaluation** — real UCI Cleveland Heart Disease data
- **DID / VC** — `did:ethr:` anchor per user at registration
- **GDPR compliance** — consent, erasure, breach notification, DPA registry

---

## Technology Stack

- **Backend**: Flask (Python 3.11+)
- **Database**: SQLAlchemy with SQLite / PostgreSQL
- **Blockchain**: Ethereum (Ganache for development, Sepolia for testnet)
- **Storage**: IPFS (Kubo)
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Smart Contracts**: Solidity `^0.8.20`
- **FL / Privacy**: scikit-learn, NumPy, custom DP + SMPC engine

---

## Prerequisites

1. **Python 3.11+**
2. **Node.js 18+** and **Truffle** (`npm install -g truffle`)
3. **Ganache** desktop or CLI (`npx ganache`)
4. **IPFS Kubo** (`ipfs daemon`)

---

## Quick Start

### Clone

```bash
git clone https://github.com/ManasSakthivel/Blockchain-Medical-Federated-Learning.git
cd blockchain-medical-fl
```

### Install dependencies

```bash
make setup
# or:  pip install -r requirements.txt && pip install pytest
```

### Configure environment

```bash
cp .env.example .env
# edit .env — set SECRET_KEY, GANACHE_URL, IPFS_URL
```

### Deploy smart contracts

```bash
truffle compile
truffle migrate --network development
```

### Run the app

```bash
make run
# → http://localhost:5000
```

Default admin credentials (created by `flask init-db`):
- Email: `admin@ehr.com`
- Password: `admin123`

---

## Tests

```bash
make test
# or:  pytest tests/ -v --tb=short
```

36 unit tests covering: Gaussian Mechanism, SMPC reconstruction, FedAvg convergence,
Byzantine detection, Flask models, auth routes, consent routes, and GDPR erasure.

---

## Benchmark

```bash
make benchmark        # full benchmark → data/benchmark_results.json
make ablation         # 4-condition ablation → data/ablation_results.json
make plot             # charts → data/plots/
```

---

## Docker (one-command stack)

```bash
make docker-up
# Services:
#   Ganache   http://localhost:7545
#   IPFS      http://localhost:5001
#   Web       http://localhost:5000
```

---

## API Routes

### Authentication
- `POST /auth/login` — user login
- `POST /auth/register` — register (DID auto-generated)
- `GET  /auth/logout` — logout

### Patient — GDPR
- `POST   /patient/erase-account` — GDPR Art. 17 erasure request
- `POST   /patient/consent/grant`  — grant consent (JSON body)
- `POST   /patient/consent/revoke` — revoke consent (JSON body)

### Patient — Medical
- `GET /patient/dashboard`
- `GET /patient/records`
- `GET /patient/lab-reports`
- `GET /patient/prescriptions`
- `GET /patient/consultations`
- `POST /patient/book-consultation`

### Doctor
- `GET /doctor/dashboard`
- `GET /doctor/patients`
- `POST /doctor/add-record/<patient_id>`
- `GET /doctor/consultations`

### Admin
- `GET /admin/dashboard`
- `GET /admin/doctors`
- `POST /admin/add-doctor`

---

## Smart Contracts

| Contract | Purpose |
|---|---|
| `EHRContract.sol` | Patient / doctor registration + medical record anchoring |
| `FederatedLearning.sol` | FL model hashes, reputation, staking, slashing |
| `ConsentContract.sol` | GDPR Art. 7 consent lifecycle with DID anchor |
| `GDPRComplianceContract.sol` | Erasure, breach notification, DPA registry |
| `FileVerificationContract.sol` | IPFS file integrity verification |
| `Roles.sol` | Role-based access control library |

All contracts use `pragma solidity ^0.8.20`.

---

## Dataset

**UCI Cleveland Heart Disease Dataset**
- Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease)
- Citation: Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology, 64(5), 304-310.
- 303 patients, 13 features, binary heart disease label
- Located at `app/processed.cleveland.data.txt`

---

## GDPR Compliance

See [`docs/GDPR_ERASURE.md`](docs/GDPR_ERASURE.md) for the full functional
erasure model, including:
- What is stored on-chain vs off-chain
- The erasure flow (Art. 17)
- DID retention as audit anchor
- Residual risks

---

## Project Structure

```
blockchain-medical-fl/
├── app/
│   ├── federated_sim_engine.py   # DP + SMPC + Cleveland data FL engine
│   ├── benchmark.py              # Real benchmark (4 conditions)
│   ├── models.py                 # DB models (User has DID field)
│   ├── routes/
│   │   ├── patient.py            # GDPR erasure + consent routes
│   │   └── ...
│   └── services/
│       └── blockchain_service.py # Web3 + GDPR/consent on-chain calls
├── contracts/
│   ├── EHRContract.sol
│   ├── FederatedLearning.sol     # Reputation + staking + slashing
│   ├── ConsentContract.sol       # NEW — GDPR Art. 7
│   ├── GDPRComplianceContract.sol # NEW — Art. 17/33/30
│   ├── FileVerificationContract.sol
│   └── Roles.sol
├── tests/
│   ├── conftest.py
│   ├── test_differential_privacy.py   # 15 unit tests
│   ├── test_federated_simulation.py   # 9 integration tests
│   └── test_app_models_routes.py      # 12 Flask + model tests
├── scripts/
│   ├── plot_convergence.py       # Charts from benchmark results
│   └── ablation_study.py        # 4-condition ablation
├── docs/
│   ├── GDPR_ERASURE.md           # Functional erasure model documentation
│   └── Blockchain-Powered Decentralized Federated Learning for Medical AI Systems.pdf
├── data/                         # Generated outputs (gitignored)
├── .github/workflows/ci.yml      # GitHub Actions CI
├── docker-compose.yml            # Ganache + IPFS + web
├── Makefile                      # Developer shortcuts
├── truffle-config.js             # Dev + Sepolia testnet config
└── requirements.txt
```

---

## Deployment to Sepolia Testnet

```bash
# Set in .env:
SEPOLIA_RPC_URL=https://rpc.sepolia.org
DEPLOYER_PRIVATE_KEY=0xYOUR_PRIVATE_KEY

truffle migrate --network sepolia
```

---

## License

MIT — see [LICENSE](LICENSE).

---

## Support

- Create a GitHub issue for bugs or feature requests
- Contact: [manas.mskg@gmail.com](mailto:manas.mskg@gmail.com)

## Acknowledgements

- UCI Machine Learning Repository — Cleveland Heart Disease dataset
- [McMahan et al. 2017] — Federated Averaging algorithm
- [Abadi et al. 2016] — Deep Learning with Differential Privacy
- [Bonawitz et al. 2017] — Practical Secure Aggregation for Privacy-Preserving ML
- Original Angular EHR version by [shamil-t](https://github.com/shamil-t/ehr-blockchain)
- Flask, Ethereum, IPFS communities
