# Blockchain Medical Federated Learning — research prototype

A Flask web application with a Solidity back-end for **consent and file-integrity
anchoring**, plus a **simulation** of federated learning (FedAvg) with client-level
differential privacy (DP) and simulated additive secret sharing on the public UCI
Cleveland Heart Disease data.

> **Status: research prototype. Not for real patient data.**
> It is not certified or assessed against HIPAA, GDPR or any other regulation, and
> nothing in this repository should be read as a compliance claim. The smart
> contracts have not been independently audited. See
> [`docs/SECURITY_AND_PRIVACY.md`](docs/SECURITY_AND_PRIVACY.md) for the exact
> threat model and limitations.

The original coursework/paper PDFs are kept in `docs/` for reference. Where they
disagree with the code or with this documentation, the code and the tests are
authoritative.

## What it does

| Area | What is implemented | What it is *not* |
|---|---|---|
| Web app | Flask, role-based access (patient / doctor / lab / admin), CSRF protection, hashed passwords, security headers, per-request authorisation helpers (`app/access.py`) | Not penetration tested; CSP still allows inline scripts |
| File handling | Uploads are encrypted at rest (AES-256-GCM, per-file key wrapped by `FILE_KEK`), served only to authorised users | Keys live in environment variables, not in an HSM/KMS |
| Blockchain | An operator account anchors an opaque id + HMAC digest of each file, consent grants/revocations, and erasure events (`contracts/`) | The chain proves *"the operator recorded X"*, not *"the patient signed X"*. No PII is written on-chain. **The app never reads the chain back**: file verification compares against the database digest, so anchors are an audit record, not a runtime tamper check |
| Consent | Database is authoritative; the chain write is best-effort and its outcome is shown honestly (`anchored` / `unavailable` / `failed`) | Revocation is not a cryptographic enforcement mechanism |
| Erasure | Account erasure deletes the database rows and **destroys the per-file keys** (crypto-shredding) | On-chain records and IPFS copies already distributed cannot be erased; they are only unlinked |
| FL simulation | FedAvg across simulated hospital nodes in **one process**; clipping + Gaussian noise; Rényi-DP accountant; Byzantine-update heuristic | Not a deployed multi-party system. SMPC here is an arithmetic simulation with no isolation |
| IPFS | `IPFSService` can add/get/unpin bytes against a Kubo node and never fabricates a CID | The upload routes do **not** currently push files to IPFS; the files stay encrypted on local disk |

## Headline experimental results (and how to read them)

Cleveland heart-disease data (n = 303, 13 features), 3 simulated nodes, 10 rounds,
5 seeds, held-out 20 % test split. Full detail and caveats:
[`docs/RESULTS.md`](docs/RESULTS.md).

| Method | Test accuracy (mean ± std, 5 seeds) | ε (δ = 1e-5) |
|---|---|---|
| Centralized (reference) | 0.846 ± 0.017 | — |
| Local-only (no federation) | 0.818 ± 0.019 | — |
| Plain FedAvg | 0.830 ± 0.020 | ∞ |
| FedAvg + DP (noise multiplier 1.5) | 0.757 ± 0.058 | 28.6 |
| FedAvg + DP + simulated SMPC (1.1) | 0.800 ± 0.046 | 43.9 |
| Full stack (0.8) | 0.833 ± 0.033 | 73.3 |

Honest reading: plain FedAvg and the lowest-noise "full stack" are about 1 percentage
point above local-only training, and DP at noise multiplier 1.5 is about 6 points below
it, but **none of the federated-vs-local differences is statistically significant**
(paired t-test over 5 seeds, |t| ≤ 2.01 against a two-sided 5 % critical value of
2.78). Only DP at noise multiplier 1.5 is significantly below the centralized reference
(t = −3.09). The ε values (28.6 to 73.3) are far too large to represent meaningful
privacy. The experiment therefore shows a working, reproducible pipeline and an honest
accountant; it does **not** demonstrate a benefit of federation or a useful
privacy/utility trade-off. Provenance and caveats: [`docs/RESULTS.md`](docs/RESULTS.md).

## Quick start (local development)

Requires Python 3.11.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt          # runtime + test tooling
cp .env.example .env                          # then fill in the values (see below)

# generate the required secrets
python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))"
python -c "import os,base64; print('FILE_KEK=' + base64.b64encode(os.urandom(32)).decode())"
python -c "import os,base64; print('ANCHOR_HMAC_KEY=' + base64.b64encode(os.urandom(32)).decode())"

export $(grep -v '^#' .env | xargs)           # or use your own env loader
export EHR_DEV_MODE=1                         # local dev: creates tables, allows http cookies
export SESSION_COOKIE_SECURE=false

flask --app run create-user --role admin --email you@example.org   # prompts for password
python run.py                                 # http://127.0.0.1:5002
```

Without `SECRET_KEY` the app refuses to start unless `EHR_DEV_MODE=1`. Without
`FILE_KEK` / `ANCHOR_HMAC_KEY` it refuses to store files (in dev mode it uses
ephemeral keys and warns, so files are unreadable after a restart). All variables
are documented in [`.env.example`](.env.example).

For anything but local development, create the schema with
`make db-upgrade` (`flask db upgrade`) and run behind a real WSGI server
(`gunicorn 'run:app'`) with TLS in front.

### Optional: local blockchain and IPFS

The app works without a chain: anchoring is reported as `unavailable`. To anchor:
run Ganache on port 7545, `npm install && npx truffle migrate --network development`,
then set `BLOCKCHAIN_RPC_URL`, `BLOCKCHAIN_OPERATOR_PRIVATE_KEY` (the deployer
account; throwaway local keys only) and, if the artifacts are not in
`build/contracts`, `CONTRACT_ARTIFACTS_DIR`. `docker-compose.yml` starts Ganache,
Kubo and the web app for local use. **These steps, and Docker, were not executed
during the audit/remediation** (no Docker, Ganache or Truffle were available); the
contracts are instead tested on an in-memory EVM, see below.

## Tests and quality gates

```bash
make test        # == python -m pytest tests -q
ruff check .     # pyflakes + syntax rules (ruff.toml)
bandit -q -r app run.py scripts -ll
pip-audit -r requirements.txt
```

At the time of writing: **184 tests pass** (security/regression tests for the web
app, 24 Solidity behaviour tests, blockchain/IPFS service tests, FL and DP tests,
migration and repository-hygiene tests). The Solidity tests run on the in-memory
`eth-tester` EVM, **not** on Ganache, and require `py-solc-x` to download solc
0.8.20 on first use. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for
exactly what has and has not been run.

## Reproduce the experiments

```bash
make benchmark   # python -m app.benchmark  -> data/benchmark_results.json, data/benchmark_summary.txt
make ablation    # python scripts/ablation_study.py
make plot        # python scripts/plot_convergence.py (needs a prior benchmark run)
```

Each result file records the git commit, library versions, seeds and exact command.
A result is only comparable if the recorded `git_commit` is clean.

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Components, data flow, trust boundaries |
| [`docs/SECURITY_AND_PRIVACY.md`](docs/SECURITY_AND_PRIVACY.md) | Threat model, controls, limitations, key handling |
| [`docs/RESEARCH_METHODOLOGY.md`](docs/RESEARCH_METHODOLOGY.md) | FL/DP/SMPC design, data handling, accountant |
| [`docs/RESULTS.md`](docs/RESULTS.md) | Benchmark numbers and their interpretation |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Environment, commands, verified vs. not verified |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Repo layout, adding routes/contracts, conventions |

## Dataset

UCI Cleveland Heart Disease (processed), 303 rows × 13 features, at
`app/processed.cleveland.data.txt` (SHA-256 `a74b7efa…427c8`). Six values are
missing (`?`) and are median-imputed from training data only. The target is
binarised (grade > 0 → 1). Detrano et al. (1989), *Am. J. Cardiology* 64(5):304–310;
source: <https://archive.ics.uci.edu/dataset/45/heart+disease>. The file was
committed by the repository author; its provenance has not been independently
re-verified against the UCI download.

`app/models/model.pkl` is a lookup table keyed by image hash used by the retinal
"classifier" demo in `app/routes/lab.py`. It is **not a trained model** and must
not be presented as one; unknown images are reported as inconclusive.

## Known limitations (summary)

- Operator-trust model: the server operator can read all decrypted data and is the only chain writer.
- `EHRContract` and `FederatedLearning` contracts are tested but **not called by the Flask app**; `FederatedLearning` is not wired to the simulation.
- The credential previously hard-coded in `scripts/fund_ganache_account.py` remains in Git history (see `docs/SECURITY_AND_PRIVACY.md`); the repository owner must confirm it is a throwaway key or rotate it.
- `npm audit` reports vulnerabilities in the Truffle development dependency tree (dev tooling only; not part of the Python runtime).
- No browser/end-to-end tests; CSP allows inline scripts.

## License

MIT — see [LICENSE](LICENSE). Original Angular EHR concept by
[shamil-t](https://github.com/shamil-t/ehr-blockchain).
