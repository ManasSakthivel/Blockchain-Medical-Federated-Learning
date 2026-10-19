# Security and privacy

This is a research prototype. **It has not been certified, audited by a third
party, or assessed against HIPAA, GDPR or any other regime, and it must not be
used with real patient data.** This document states what the code actually does,
what it does not do, and where the trust boundaries are. Every claim below was
checked against the code or a test; the file/test references are given.

## 1. Threat model

| Actor | Assumed capability | Defended against? |
|---|---|---|
| Anonymous internet user | HTTP requests | Yes: login required, CSRF tokens, login rate-limited at 10/min per IP (Flask-Limiter; see §2 and §7 for the per-process caveat) |
| Authenticated user of another role | Cross-role / cross-patient requests (IDOR) | Yes, enforced centrally in `app/access.py`; tests in `tests/test_security_regressions.py` |
| Malicious doctor / lab | Valid account | Partly: a doctor only reaches patients with an *accepted* consultation (`TREATING_STATUSES`); a doctor who has a treating relationship can read that patient's data |
| Passive reader of the database or upload directory | Copy of DB / disk | Partly: uploaded files are ciphertext; the database itself (names, clinical text) is **not** encrypted at the field level |
| Reader of the public blockchain | Sees all on-chain data | Yes by design: only opaque ids, keyed digests and pseudonymous addresses are written |
| **Server operator / anyone with the process environment** | Has `FILE_KEK`, `ANCHOR_HMAC_KEY`, the operator chain key, and DB | **No.** The operator is trusted. |
| Malicious FL participant | Submits poisoned updates | Heuristic detection only; no guarantee (see §6) |

## 2. Application controls (implemented and tested)

| Control | Where | Test |
|---|---|---|
| App refuses to start without `SECRET_KEY` (placeholders rejected) unless `EHR_DEV_MODE=1` | `app/config.py:28-40` | `test_security_regressions.py` (config tests) |
| Secure/HttpOnly/SameSite cookies; `Secure` is on by default | `app/config.py:50-55` | same |
| CSRF on every state-changing request (forms: hidden token; `fetch`: `X-CSRFToken` added in `static/js/main.js`) | `app/__init__.py`, templates | same |
| Server-side sessions are `id:token`; token rotates on logout, password change and erasure, so stale cookies stop working | `app/__init__.py:38-50`, `app/models.py` | same |
| Password hashing via Werkzeug (scrypt/pbkdf2 default), min length 10, constant-time-ish login (dummy hash for unknown users) | `app/access.py:19-33`, `app/routes/auth.py:18,53` | same |
| Self-registration limited to `patient` and `doctor`; `admin`/`lab` can only be created by an operator (`flask --app run create-user`) | `app/routes/auth.py:15,80`, `run.py` | `test_ops_hygiene.py` |
| Admins have **no** access to PHI routes; admin routes are role-gated | `app/routes/admin.py` | same |
| Object-level authorisation for patients, doctors, labs, files, lab requests | `app/access.py`, routes | same |
| Uploads: extension allow-list, size cap (16 MB), content checks, random storage names, never served from `static/` | `app/services/file_store.py` | same |
| Uploaded files encrypted with AES-256-GCM; per-file random data key wrapped by `FILE_KEK`; ciphertext bound to file id via AAD | `app/services/crypto_service.py` | same |
| Security headers: CSP, `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cache-Control: no-store` for non-static responses, HSTS when cookies are secure | `app/__init__.py:90-121` | same |
| Third-party CSS/JS loaded with Subresource Integrity | `app/templates/base.html` | manual |
| Error handlers return generic messages and log details server-side | `app/__init__.py:124-156` | same |
| `run.py` never enables the debugger or binds publicly by default | `run.py` | `test_ops_hygiene.py` |

## 3. Blockchain: what is and is not protected

**Trust model.** A single operator account (`BLOCKCHAIN_OPERATOR_PRIVATE_KEY`,
the contract deployer) is the only account allowed to write to
`ConsentContract`, `FileVerificationContract` and `ErasureLogContract`
(`onlyOperator`, `contracts/*.sol`; tests: `tests/test_contracts.py`, with
mutation checks that removing the guard makes tests fail). Therefore the chain
proves **"the operator recorded X at block N"**. It does **not** prove that a
patient signed or authorised X, and it is not a decentralised trust anchor.

**What is written on-chain** (and nothing else):

- `FileVerificationContract.anchorFile(bytes32 publicId, bytes32 digest)` — a random public id and `HMAC-SHA256(ANCHOR_HMAC_KEY, …)` of the file. Without the HMAC key a third party cannot confirm a guessed file against the digest.
- `ConsentContract.grantConsent(patient, recipient, dataType, purposeHash)` — `patient` is the pseudonymous address derived from the user's `did:ethr:` identifier; `purposeHash` is `HMAC-SHA256(ANCHOR_HMAC_KEY, "purpose:" + purpose)`. Using a keyed HMAC (same key as file anchor digests, different domain prefix) prevents dictionary attacks on low-entropy purpose strings such as `"treatment"` or `"research"` on a public ledger. **Legacy note:** rows written before this change carry plain SHA-256 digests; new rows carry HMAC digests. The application never re-verifies the stored hash at runtime, so both coexist safely in the DB.
- `ErasureLogContract.recordErasure(bytes32 ref)` — a random reference unrelated to any person.

**Pseudonymous DID.** At registration `Account.create()` generates a fresh key and
only the address is kept (`app/routes/auth.py:123-128`). The user does not control
that key and cannot sign with it. It is a pseudonym, not a self-sovereign identity.
It is nulled on erasure (`app/routes/patient.py:~460`) but stays on-chain in any
consent events already written (unlinked from the person once the DB row is gone).

**The chain is write-only from the application's point of view.** No route reads consent or file anchors back (`BlockchainService.get_file_anchor` and `has_consent` exist and are tested, but are not called by the app). Integrity verification compares against the digest stored in the database, so the on-chain anchor does not currently strengthen the runtime check; it is an externally checkable audit record. Do not describe the system as providing blockchain-backed tamper-proofing.

**Unused contracts.** `EHRContract` and `FederatedLearning` are compiled and tested
but **not called by the Flask app or the simulation**. They should be read as
specified-and-tested prototypes, not as part of a running system.

**Not done:** the contracts have not been formally verified or externally audited;
no public-network deployment config is provided (`truffle-config.js` has a local
development network only).

## 4. Consent and erasure — exact behaviour

Consent (`app/routes/patient.py:339-379`): the **database is authoritative**.
Granting writes the row first, then tries to anchor; the response and the stored
`chain_status` are `anchored`, `unavailable` or `failed`. Revocation is committed to
the database before any chain call, so it takes effect immediately even if the
chain is down. The chain record is an audit trail, not an enforcement point;
nothing outside this server reads it.

Erasure (`POST /patient/erase-account`, password-confirmed,
`app/routes/patient.py:~399`):

- Deletes the patient's clinical rows, consultations, lab requests/reports, consents and verification logs.
- **Crypto-shreds** each stored file: the wrapped data key is destroyed and the ciphertext deleted, so any retained copy of the ciphertext is undecryptable.
- Overwrites personal fields, nulls the DID, disables login, rotates the session token.
- Writes an `ErasureLog` row and (best-effort) an opaque on-chain erasure reference.
- Attempts to unpin any recorded IPFS CID and reports `ipfs_unpinned` / `ipfs_unpin_failed`.

What it does **not** do, and what must not be claimed:

- It does **not erase immutable blockchain data**. On-chain consent/anchor records remain; they hold no names or e-mails, but they are *unlinked*, not erased.
- It cannot recall copies already pinned or replicated by other IPFS nodes. (The current upload routes do not publish to IPFS; `ipfs_status` stays `not_attempted`. The unpin path exists for future use and is covered by service-level tests only.)
- Database backups, logs and any exported data are out of scope.
- It is **not** a GDPR Article 17 compliance mechanism; legal erasure obligations depend on the controller's whole process.

**ErasureLog privacy trade-off.** The `ErasureLog` row records only a random `erasure_ref` (opaque bytes), the `requested_at` timestamp, and the erasing user's id (stored on the application side before the user row is overwritten). It intentionally omits the name, e-mail, and any PHI of the erased account to avoid the log itself becoming a privacy liability. This means: (a) after erasure, no audit query can attribute the log entry back to the former user; the `erasure_ref` is the only receipt; (b) `requested_at` is a weak metadata channel (approximate time of erasure is visible to anyone who can read the log table), but no identity is linked; (c) the on-chain `ErasureLogContract.recordErasure` record is similarly opaque. Do not describe the ErasureLog as an audit trail for the *identity* of the erased user — it is an audit trail for the *event of erasure*.

## 5. Secrets and credentials

- No secret has a usable default. `SECRET_KEY`, `FILE_KEK`, `ANCHOR_HMAC_KEY` and the operator key are read from the environment. `.env.example` contains no values; `tests/test_ops_hygiene.py` fails if a hex private key is assigned in tracked text files.
- **`FILE_KEK` rotation is not implemented.** Changing it makes existing files unreadable. The `kek_id` field exists for a future rotation scheme.
- **`ANCHOR_HMAC_KEY` must remain constant across restarts and deployments.** It is used as the key in `HMAC-SHA256` for file anchor digests (`anchorFile`), consent purpose hashes (`grantConsent`), and user pseudonyms. Changing it silently invalidates every stored `anchor_digest`, `pseudonym`, and `purpose_hmac` value in the database, and any future on-chain verification will fail. There is no rotation mechanism; treat it as a long-lived symmetric secret and protect it accordingly.
- **Historical exposure (action required by the repository owner).** Earlier commits contained a 64-hex value assigned as a private key in `scripts/fund_ganache_account.py` (first present in commit `d5143df`), additional unexplained 64-hex strings in the old `.env.example`, `docs/IMPLEMENTATION_SUMMARY.md` and `scripts/setup_ganache.py`, and the well-known public Hardhat/Anvil test keys and mnemonic. The values were removed from the working tree but **remain in Git history**; history was deliberately not rewritten. A read-only check of the address derived from the first value found zero balance and zero transactions on Ethereum mainnet, Sepolia, Polygon and BSC on the audit date, but its origin is unknown. The owner must confirm it is a throwaway key, or treat it as compromised and rotate it. The Hardhat/Anvil keys are public and must never hold value.
- The Ganache service in `docker-compose.yml` uses `--deterministic`, i.e. a **publicly known** mnemonic. It is for local use only, and ports are bound to `127.0.0.1`.

## 6. Federated-learning privacy claims

See [`RESEARCH_METHODOLOGY.md`](RESEARCH_METHODOLOGY.md). In short:

- Client-level Gaussian-mechanism DP with a Rényi accountant is implemented, but the ε values obtained (28.6 to 73.3 at δ=1e-5) are **too large to constitute meaningful privacy**.
- Noise comes from `numpy.random.Generator`, which is **not a cryptographically secure RNG**.
- The "SMPC" is additive secret sharing executed inside one Python process. It demonstrates the arithmetic and adds **no** privacy against the process that runs it. Do not describe it as "the aggregator never sees raw gradients".
- Byzantine detection is a heuristic (update norm > 5× the median, needs ≥ 3 clients). It is informational and offers no robustness guarantee; an attacker can stay under the threshold.
- No FL data or model hash is currently written to a chain by the application.

## 7. Known limitations and residual risks

| Item | Risk | Status |
|---|---|---|
| CSP allows `'unsafe-inline'` scripts (inline `<script>` blocks remain) | Reduces XSS defence in depth | Open; user data is auto-escaped by Jinja and inline handlers no longer interpolate data |
| Login rate-limited at 10 requests/minute per IP (`@limiter.limit` on `POST /auth/login`, `app/routes/auth.py`); no account lockout; in multi-worker deployments the per-process default `memory://` storage means the limit is per-process, not global — set `RATELIMIT_STORAGE_URI` to a shared Redis or Memcached URI to enforce the limit across workers | Online password guessing — partially mitigated; per-process caveat applies in production | Implemented; production caveat documented in `.env.example` |
| No MFA, no password-reset flow | Account takeover recovery | Open (out of scope for prototype) |
| Database fields (names, clinical text) not encrypted at rest | Disk/DB leak exposes PHI | Open; use disk/DB encryption in any deployment |
| Single `FILE_KEK`, in an environment variable; no rotation/KMS | Key compromise = all files | Open |
| SQLite default | Not suited to concurrent production use | Documented |
| Truffle dev dependency tree has many `npm audit` advisories | Dev tooling only; not shipped | Open; Python deps pass `pip-audit` |
| `app/models/model.pkl` is loaded with `pickle` | Unsafe if the file is ever replaced by an untrusted one | Mitigated: repository asset only, never user-supplied (`app/routes/lab.py:34-52`) |
| Chain anchoring is best-effort and unauthenticated by the patient | Weak non-repudiation | By design; stated in §3 |
| No browser-level or penetration testing | Unknown client-side issues | Not run |

## 8. Reporting

Report vulnerabilities by opening a private security advisory on the GitHub
repository, or by e-mail to the address in the repository profile.
