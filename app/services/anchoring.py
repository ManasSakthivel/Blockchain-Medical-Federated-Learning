"""Best-effort blockchain anchoring facade.

The database is authoritative. Anchoring never blocks or fails a request: it
returns an explicit status so the UI and the API can say what really happened.
``anchored``  - a mined transaction exists (receipt status 1)
``unavailable`` - no chain configured / reachable; nothing was attempted
``failed``    - a transaction was attempted and reverted or errored
"""
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class AnchorOutcome:
    status: str = "unavailable"
    tx_hash: str | None = None
    object_id: int | None = None
    match: bool | None = None


def _service():
    try:
        from app.services.blockchain_service import BlockchainService
        return BlockchainService.from_env()
    except Exception:  # not configured / web3 missing: treated as unavailable
        log.info("blockchain service unavailable", exc_info=True)
        return None


def anchor_file(public_id: str, anchor_digest: str) -> AnchorOutcome:
    svc = _service()
    if svc is None:
        return AnchorOutcome()
    try:
        res = svc.anchor_file(public_id, anchor_digest)
    except Exception:
        log.exception("anchor_file failed")
        return AnchorOutcome(status="failed")
    return AnchorOutcome(res.status, res.tx_hash, res.object_id)


def record_verification(chain_file_id: int | None, is_match: bool) -> AnchorOutcome:
    svc = _service()
    if svc is None or chain_file_id is None:
        return AnchorOutcome()
    try:
        res = svc.record_file_verification(chain_file_id, is_match)
    except Exception:
        log.exception("record_file_verification failed")
        return AnchorOutcome(status="failed")
    return AnchorOutcome(res.status, res.tx_hash, res.object_id)


def anchor_consent(patient_id: int, did: str | None, recipient: str, data_type: str,
                   purpose_hash: str) -> AnchorOutcome:
    """Record a consent grant on-chain. ``did`` is the pseudonymous identifier,
    never a name/e-mail. ``patient_id`` is accepted for logging only."""
    svc = _service()
    if svc is None:
        return AnchorOutcome()
    try:
        res = svc.grant_consent(did or "", recipient, data_type, purpose_hash)
    except Exception:
        log.exception("grant_consent failed (patient_id=%s)", patient_id)
        return AnchorOutcome(status="failed")
    return AnchorOutcome(res.status, res.tx_hash, res.object_id)


def revoke_consent_anchor(chain_consent_id: int | None) -> AnchorOutcome:
    svc = _service()
    if svc is None or chain_consent_id is None:
        return AnchorOutcome()
    try:
        res = svc.revoke_consent(chain_consent_id)
    except Exception:
        log.exception("revoke_consent failed")
        return AnchorOutcome(status="failed")
    return AnchorOutcome(res.status, res.tx_hash, res.object_id)


def record_erasure(erasure_ref: str) -> AnchorOutcome:
    """Record that *an* erasure happened, keyed by an opaque random reference
    that is not linked to any person."""
    svc = _service()
    if svc is None:
        return AnchorOutcome()
    try:
        res = svc.record_erasure(erasure_ref)
    except Exception:
        log.exception("record_erasure failed")
        return AnchorOutcome(status="failed")
    return AnchorOutcome(res.status, res.tx_hash, res.object_id)


def unpin(cid: str) -> bool:
    """Unpin a CID from the configured IPFS node. False when it could not be
    confirmed (node unreachable / not configured)."""
    try:
        from app.services.ipfs_service import IPFSService
        return bool(IPFSService.from_env().unpin(cid))
    except Exception:
        log.info("ipfs unpin unavailable", exc_info=True)
        return False
