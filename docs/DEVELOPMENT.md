# Development guide

## Layout

```
app/
  __init__.py            create_app(): config, CSRF, login, blueprints, headers, error handlers
  config.py              build_config(): fail-closed secrets, cookie flags, upload dir
  access.py              roles_required, ownership/treatment checks, password/email validation
  models.py              SQLAlchemy models
  routes/                auth, patient, doctor, lab, admin, file_verification, main
  services/              crypto_service, file_store, anchoring, blockchain_service, ipfs_service
  federated_sim_engine.py, benchmark.py   FL / DP / SMPC simulation
  templates/, static/
contracts/               Solidity 0.8.20 (Consent, FileVerification, ErasureLog, EHR, FederatedLearning, Roles, Migrations)
migrations/              Truffle deployment scripts (JS)
db_migrations/           Alembic/Flask-Migrate schema migrations (Python)
scripts/                 ablation_study.py, paired_stats.py, plot_convergence.py, one-off DB helpers
tests/                   pytest suite
docs/                    this documentation (+ the original PDFs)
```

Note the two `migrations` directories: `migrations/` is Truffle, `db_migrations/` is
the database.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-test.txt
export SECRET_KEY=$(python -c "import secrets;print(secrets.token_hex(32))")
export EHR_DEV_MODE=1 SESSION_COOKIE_SECURE=false
python run.py              # http://127.0.0.1:5002
```

`EHR_DEV_MODE=1` creates the tables with `db.create_all()` and uses ephemeral
encryption keys (files become unreadable after a restart). Outside dev mode run
`make db-upgrade` and provide `FILE_KEK` / `ANCHOR_HMAC_KEY`
(`.env.example`). Create accounts with
`flask --app run create-user --role {admin|doctor|patient|lab} --email ...`
(prompts for the password). Doctor and lab accounts created this way have a `User`
row only; their profile rows (`Doctor`/`Lab`) are created through the registration
form or a shell session, and routes tolerate a missing profile by redirecting.

## Tests and checks

```bash
python -m pytest tests -q          # also: make test
ruff check .
bandit -q -r app run.py scripts -ll
pip-audit -r requirements.txt
```

- Fixtures (`tests/conftest.py`): `app` (in-memory SQLite, ephemeral keys, no chain), `client`, `make` (factory for users, patients, doctors, labs, consultations, lab requests), `login_as(user)`.
- CSRF is disabled in the `app` fixture except in tests that enable it explicitly.
- Add a regression test for every security fix. Cross-role / cross-patient access tests belong in `tests/test_security_regressions.py`.
- Contract tests compile with `py-solc-x` and execute on eth-tester; run them with `-k contracts`.

## Conventions

- **Authorisation:** decorate with `@roles_required(...)`; for patient data go through `app/access.py` (`doctor_has_access`, `can_access_file`, `can_upload_for_patient`). Never accept a patient id from the client without checking it against the caller.
- **State-changing routes** must be `POST`/`PUT`/`DELETE` and include a CSRF token (forms: hidden `csrf_token`; `fetch`: handled by `static/js/main.js`).
- **Templates:** never interpolate data into inline JavaScript. Use `data-*` attributes and read them in script.
- **Secrets:** read from the environment; no defaults; no logging of values. Do not commit `.env`, keys, `*.db` or uploads.
- **Chain calls:** go through `app/services/anchoring.py`, commit the database first, store the returned `status`, never let a chain failure fail the request, and never write personal data on-chain.
- **FL results:** never hard-code numbers; always regenerate from `app/benchmark.py`; report negative results.
- **Dependencies:** pin in `requirements.txt`; run `pip-audit` after changes.

## Adding things

- *A route:* add it to the relevant blueprint with `@roles_required`, write the happy-path and a cross-user denial test.
- *A model change:* edit `app/models.py`, then `FLASK_APP=run.py flask db migrate --directory db_migrations -m "..."`, review the generated script, run `flask db upgrade` and `flask db check` (the `test_ops_hygiene` migration test fails on drift).
- *A contract change:* edit `contracts/`, extend `tests/test_contracts.py`, update `migrations/2_deploy_contracts.js` and the matching method in `app/services/blockchain_service.py`.

## Local blockchain and containers

Not exercised during the remediation (no Docker, Ganache or Truffle available):

```bash
npm install
npx ganache --deterministic --port 7545 --chain.chainId 1337     # public mnemonic: dev only
npx truffle migrate --network development
export BLOCKCHAIN_RPC_URL=http://127.0.0.1:7545
export BLOCKCHAIN_OPERATOR_PRIVATE_KEY=<the deploying account's key>
docker compose up --build        # alternative: needs a filled .env
```

`BLOCKCHAIN_OPERATOR_PRIVATE_KEY` must belong to the account that ran the migration,
because that account is the contracts' operator.

## Known cleanup candidates

- `scripts/migrate_lab_request.py`, `scripts/migrate_lab_tables.py` and `scripts/fix_patient_profiles.py` are one-off helpers from before Alembic was set up; they are not used by the app or the tests. (`app/federated_node.py` and `app/heart_disease_data.py` were removed: unreferenced, depended on the uninstalled `flwr`, and the loader fitted its scaler on the full dataset before splitting.)
- Wire `FederatedLearning` to the simulation or remove it; same for `EHRContract`.
- Call `IPFSService.add_bytes` from the upload path (ciphertext only) or remove the IPFS claims.
- Add login rate limiting, a lock file, and a CSP without `'unsafe-inline'`.
