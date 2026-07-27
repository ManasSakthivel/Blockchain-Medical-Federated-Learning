# GDPR Erasure — Functional Erasure Model

## Overview

This document describes how the **Blockchain Medical Federated Learning** system
implements GDPR Article 17 ("Right to Erasure") in the context of a blockchain
where on-chain data is immutable by design.

---

## The Tension: GDPR Art. 17 vs Blockchain Immutability

Blockchain ledgers are append-only. Once a transaction is mined, it cannot be
deleted. A naïve implementation that stores raw PII on-chain would be
fundamentally incompatible with GDPR.

This system resolves the tension with a **functional erasure model** inspired by
the approach endorsed by the UK ICO and the European Data Protection Board (EDPB):

> *"Rendering personal data permanently inaccessible counts as erasure, even if
> the underlying cryptographic record persists on the ledger."*
> — EDPB Guidelines 05/2022 on Personal Data in Blockchain

---

## What We Store On-Chain

The system stores **no raw PII** on the Ethereum blockchain. Instead:

| On-chain | Off-chain |
|---|---|
| SHA-256 hash of medical record | Full record content (SQLite / PostgreSQL) |
| IPFS CID of consent statement | Consent metadata |
| DID anchor (`did:ethr:<address>`) | Name, email, DOB, address |
| Erasure event (emitted as EVM log) | ← nothing — the event is the proof |

---

## Erasure Flow (Art. 17)

### 1. Patient requests erasure

`DELETE /patient/erase-account` (or `POST` with `_method=DELETE`)

### 2. Off-chain erasure (immediate, guaranteed)

The following personal data is **immediately nullified** in the relational database:

| Table | Action |
|---|---|
| `MedicalRecord` | All rows for this patient **deleted** |
| `Prescription` | All rows deleted |
| `LabReport` | All rows deleted |
| `LabRequest` | All rows deleted |
| `Consultation` | All rows deleted |
| `Patient` | PII columns set to `[ERASED]` / `null` |
| `User` | `email` and `username` replaced with random token; `password_hash` wiped |

The `did` column is intentionally **retained** as an audit anchor (see below).

### 3. IPFS erasure (best-effort)

Files pinned to IPFS are unpinned from the local node. They may persist in
the IPFS network if other nodes pinned them — this is addressed by the
"data controller" responsibility model: the system cannot guarantee third-party
IPFS node behaviour.

### 4. On-chain audit event (best-effort)

`GDPRComplianceContract.requestErasure(ipfsCid)` is called, which emits:

```solidity
event ErasureRequested(
    address indexed subject,
    string  ipfsCidToErase,
    uint256 timestamp
);
```

This provides a timestamped, tamper-proof **proof of intent to erase** — exactly
what a DPA audit would require under Art. 5(2) (accountability principle).

### 5. Session invalidated

The user is logged out immediately. Their account cannot be used again.

---

## The "Functional Erasure" Argument

The on-chain record hashes become **cryptographically orphaned**:

- The pre-image (raw medical data) has been deleted → the hash is meaningless
- The DID is retained but links to an anonymised, inaccessible account
- No one — including the data controller — can reconstruct the original data
  from what remains on-chain

This satisfies the EDPB's "equivalence of effect" test for erasure.

---

## DID Retention as Audit Anchor

The `did:ethr:<address>` DID is intentionally **not erased** because:

1. It contains no PII (it is a pseudonymous Ethereum address, not a name or email)
2. It anchors the erasure event on-chain to a specific identity
3. Regulators can verify *that* erasure occurred without accessing *what* was erased

This is consistent with Recital 26 of GDPR:
> *"Pseudonymous data, which could be attributed to a natural person by the use
> of additional information, should be considered to be information on an
> identifiable natural person."*

Since the "additional information" (the mapping from DID to real identity)
has been deleted, the DID no longer constitutes personal data post-erasure.

---

## GDPR Articles Addressed

| Article | Mechanism |
|---|---|
| Art. 7 — Consent | `ConsentContract.sol` + `/patient/consent/grant` / `revoke` |
| Art. 17 — Right to Erasure | `DELETE /patient/erase-account` (this document) |
| Art. 20 — Data Portability | `ConsentContract.requestDataPortability()` event |
| Art. 25 — Privacy by Design | DP + SMPC — no raw data leaves nodes |
| Art. 30 — Records of Processing | `GDPRComplianceContract` audit trail |
| Art. 33 — Breach Notification | `GDPRComplianceContract.notifyBreach()` |

---

## Limitations and Residual Risks

1. **IPFS persistence** — data may persist on third-party IPFS nodes.
2. **Blockchain logs** — EVM event logs are immutable. The *hash* of the
   pre-image persists; the pre-image itself does not.
3. **Backup media** — database backups must be rotated per data retention policy.
4. **Legal basis for DID retention** — must be documented in the Record of
   Processing Activities (ROPA) per Art. 30.

---

*This document is maintained as part of the Data Protection Impact Assessment
(DPIA) for the Blockchain Medical FL system.*
