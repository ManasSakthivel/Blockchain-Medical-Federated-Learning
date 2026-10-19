# Reproducibility

## 1. Environment actually used

| Item | Value |
|---|---|
| OS | macOS (arm64) |
| Python | 3.11.15 |
| Key packages | Flask 3.1.3, Werkzeug 3.1.9, SQLAlchemy 2.0.55, web3 7.16.0, eth-tester 0.14.0b1, py-evm 0.12.1b1, py-solc-x 2.0.5 (solc 0.8.20), cryptography 50.0.2, numpy 1.25.2, scikit-learn 1.5.2, scipy 1.16.3, pandas 2.1.0, pytest 9.1.1, ruff 0.17.1, bandit 1.9.4 |
| Not available | Docker, Ganache, Truffle/Node tooling for contracts, a Kubo IPFS node |

`requirements.txt` pins direct runtime dependencies (not transitive ones; there is
no lock file). Resolving the same versions on another platform may differ in
transitive packages. `numpy==1.25.2` and `pandas==2.1.0` are old; they were kept
because the benchmark numbers were produced with them. `pip-audit -r requirements.txt`
reported no known vulnerabilities on the run date.

## 2. Commands and outcomes

Status key: **PASS** ran and succeeded; **FAIL**; **NOT RUN** not attempted;
**BLOCKED** could not be run in this environment.

| Check | Command | Status |
|---|---|---|
| Full test suite | `DATABASE_URL=sqlite:///:memory: SECRET_KEY=test python -m pytest tests -q` | **PASS** — 184 passed |
| Web app + models | `tests/test_app_models_routes.py` (19), `tests/test_security_regressions.py` (47) | PASS |
| Solidity behaviour | `tests/test_contracts.py` (24) on the in-memory eth-tester EVM — **not Ganache** | PASS |
| Chain / IPFS services | `tests/test_blockchain_service.py` (23), against eth-tester and fake HTTP sessions | PASS |
| FL / DP | `tests/test_differential_privacy.py` (18), `test_federated_simulation.py` (10), `test_fl_regression.py` (35) | PASS |
| Ops / hygiene | `tests/test_ops_hygiene.py` (8): CLI, migration, no hard-coded keys, compose | PASS |
| Lint | `ruff check .` (rules E9, F per `ruff.toml`) | PASS |
| Static security | `bandit -q -r app run.py scripts -ll` | PASS — no medium/high findings; 4 low (subprocess/pickle) accepted |
| Dependency audit (Python) | `pip-audit -r requirements.txt` | PASS — no known vulnerabilities |
| Migrations | `flask db upgrade`, `flask db check`, `flask db downgrade` on empty SQLite | PASS (SQLite only; PostgreSQL **NOT RUN**) |
| Benchmark | `python -m app.benchmark` and `python scripts/ablation_study.py` | PASS (runs); numbers in [`RESULTS.md`](RESULTS.md) |
| Paired statistics | `python scripts/paired_stats.py` | PASS |
| `make plot` | `scripts/plot_convergence.py` | **NOT RUN** |
| Truffle compile / migrate | `npx truffle compile && npm run migrate` | **BLOCKED** — no Truffle/Ganache. The contracts were compiled with solc 0.8.20 via py-solc-x instead |
| Contracts on Ganache | — | **BLOCKED** |
| Docker / compose stack | `docker compose up` | **BLOCKED** — no Docker |
| Real IPFS node | — | **BLOCKED** |
| Browser / end-to-end tests | — | **NOT RUN** (none exist) |
| `npm audit` (dev tooling) | `npm audit --package-lock-only` | Ran: 117 advisories (18 low, 53 moderate, 37 high, 9 critical) in the Truffle dev-dependency tree; not remediated |
| Third-party penetration test / contract audit | — | **NOT RUN** |
| Independent DP accountant cross-check | — | **NOT RUN** |

The mutation checks on the contract tests (removing the operator guard, the
participant authorisation, the consent-gate-unset check and the original-doctor
check) each made tests fail, which indicates those tests are effective. They were
run on a scratch copy and are not part of the repository.

## 3. Regenerating the results

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
python -m app.benchmark                  # data/benchmark_results.json + benchmark_summary.txt
python scripts/paired_stats.py           # paired t statistics from that JSON
python scripts/ablation_study.py         # data/ablation_results.json (single seed)
```

- Determinism: every random source is seeded from `--seed` (`numpy.random.default_rng`, scikit-learn `random_state`, and the legacy `np.random.seed` for the attacker noise). Re-running with the same seeds, the same package versions and the same data file gives the same numbers (checked by running the full benchmark twice and diffing the JSON, excluding timings: identical).
- Results depend on the pinned library versions (notably scikit-learn's lbfgs solver). Different versions may change the last digit or two.
- `metadata.git_commit` and `metadata.git_dirty` identify the code that produced a file. The `data/benchmark_results.json` shipped with a commit is a snapshot; regenerate before citing.
- Runtime: under one second for the benchmark; a few seconds for the full test suite (about 16 s).

## 4. Running the contract tests

`tests/test_contracts.py` and `tests/test_blockchain_service.py` use `py-solc-x`,
which downloads solc 0.8.20 on first use (network access required once). If solc 0.8.20
cannot be obtained the Solidity tests are **skipped** (pytest reports them as
skipped, so check the summary line for `skipped` before trusting a green run); they
never fall back to a mock. In CI the compiler is downloaded on the runner.

## 5. Known reproducibility gaps

- No lock file; transitive versions float.
- Dataset provenance: the committed Cleveland file's SHA-256 is recorded but was not compared with a fresh UCI download.
- Cross-platform floating-point differences are possible (verified on macOS arm64 only).
- Truffle/Ganache/Docker flows are documented but unexecuted; treat them as untested.
