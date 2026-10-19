"""Operator-signed blockchain anchoring client (web3 7.x).

Trust model
-----------
Exactly one account, the *operator* (``BLOCKCHAIN_OPERATOR_PRIVATE_KEY``), signs
every transaction. A mined transaction therefore proves "the operator recorded
this", not "the patient/doctor signed this". The application database stays
authoritative for access decisions.

Only opaque identifiers and keyed digests are sent (see contracts/*.sol). The
operator key is read from the environment only: never from a file in the repo,
never logged, never returned.

Every public method returns a :class:`ChainResult`; none of them raises for
chain problems, so callers can degrade gracefully and report the real status:
``anchored`` (receipt status 1), ``unavailable`` (nothing configured / reachable)
or ``failed`` (attempted, reverted or errored).
"""
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

CONTRACT_NAMES = ("ConsentContract", "FileVerificationContract", "ErasureLogContract")
_ENV_ADDR = {
    "ConsentContract": "CONSENT_CONTRACT_ADDRESS",
    "FileVerificationContract": "FILE_VERIFICATION_CONTRACT_ADDRESS",
    "ErasureLogContract": "ERASURE_LOG_CONTRACT_ADDRESS",
}


class BlockchainUnavailable(RuntimeError):
    """Chain is not configured or not reachable."""


@dataclass
class ChainResult:
    status: str  # anchored | unavailable | failed
    tx_hash: str | None = None
    object_id: int | None = None
    error: str | None = None
    match: bool | None = None

    @property
    def ok(self) -> bool:
        return self.status == "anchored"


def _bytes32(value: str, *, name: str) -> bytes:
    """Hex string (optionally 0x-prefixed, <= 32 bytes) -> bytes32 (right-padded)."""
    raw = value[2:] if value.startswith(("0x", "0X")) else value
    try:
        data = bytes.fromhex(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be hex") from exc
    if not 0 < len(data) <= 32:
        raise ValueError(f"{name} must be 1..32 bytes")
    return data.ljust(32, b"\x00")


def address_from_did(did: str) -> str:
    """``did:ethr:0x...`` -> checksum address (the throw-away public address)."""
    from web3 import Web3

    addr = (did or "").rsplit(":", 1)[-1]
    if not Web3.is_address(addr):
        raise ValueError("DID does not contain a valid address")
    return Web3.to_checksum_address(addr)


class BlockchainService:
    def __init__(self, web3, contracts: dict, operator_private_key: str, *,
                 receipt_timeout: int = 60):
        self.w3 = web3
        self.contracts = contracts
        self._key = operator_private_key  # never logged / exposed
        self.operator = web3.eth.account.from_key(operator_private_key).address
        self.receipt_timeout = receipt_timeout

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_env(cls) -> "BlockchainService":
        """Build from environment. Raises BlockchainUnavailable when not configured."""
        from web3 import Web3

        rpc = os.environ.get("BLOCKCHAIN_RPC_URL")
        key = os.environ.get("BLOCKCHAIN_OPERATOR_PRIVATE_KEY")
        if not rpc or not key:
            raise BlockchainUnavailable("BLOCKCHAIN_RPC_URL / BLOCKCHAIN_OPERATOR_PRIVATE_KEY not set")
        w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 10}))
        if not w3.is_connected():
            raise BlockchainUnavailable("RPC endpoint is not reachable")
        artifacts = Path(os.environ.get("CONTRACT_ARTIFACTS_DIR", "build/contracts"))
        contracts = {}
        for name in CONTRACT_NAMES:
            contracts[name] = cls._load_contract(w3, name, artifacts)
        return cls(w3, contracts, key)

    @staticmethod
    def _load_contract(w3, name: str, artifacts: Path):
        path = artifacts / f"{name}.json"
        if not path.exists():
            raise BlockchainUnavailable(f"missing artifact {path}")
        art = json.loads(path.read_text())
        addr = os.environ.get(_ENV_ADDR[name])
        if not addr:
            nets = art.get("networks", {})
            addr = (nets.get(str(w3.eth.chain_id)) or {}).get("address")
        if not addr:
            raise BlockchainUnavailable(f"{name} is not deployed on chain {w3.eth.chain_id}")
        return w3.eth.contract(address=w3.to_checksum_address(addr), abi=art["abi"])

    def status(self) -> dict:
        """Non-secret health summary."""
        try:
            return {"connected": bool(self.w3.is_connected()), "chain_id": self.w3.eth.chain_id,
                    "operator": self.operator,
                    "contracts": {n: c.address for n, c in self.contracts.items()}}
        except Exception as exc:
            return {"connected": False, "error": type(exc).__name__}

    # --------------------------------------------------------------- plumbing
    def _send(self, fn, event_name: str | None = None, contract=None) -> ChainResult:
        try:
            tx = fn.build_transaction({
                "from": self.operator,
                "nonce": self.w3.eth.get_transaction_count(self.operator, "pending"),
                "chainId": self.w3.eth.chain_id,
                "gasPrice": self.w3.eth.gas_price,
            })
            signed = self.w3.eth.account.sign_transaction(tx, private_key=self._key)
            tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=self.receipt_timeout)
        except BlockchainUnavailable as exc:
            return ChainResult("unavailable", error=str(exc))
        except Exception as exc:  # revert at estimate time, RPC error, timeout ...
            log.warning("chain transaction failed: %s", type(exc).__name__)
            return ChainResult("failed", error=f"{type(exc).__name__}: {exc}"[:300])

        tx_hex = "0x" + bytes(tx_hash).hex()
        if receipt["status"] != 1:
            return ChainResult("failed", tx_hash=tx_hex, error="transaction reverted")
        object_id = None
        if event_name:
            events = getattr(contract.events, event_name)().process_receipt(receipt)
            if events:
                args = events[0]["args"]
                object_id = int(args.get("fileId", args.get("consentId", 0)) or 0) or None
        return ChainResult("anchored", tx_hash=tx_hex, object_id=object_id)

    @staticmethod
    def _guard(call):
        try:
            return call()
        except ValueError as exc:
            return ChainResult("failed", error=f"invalid input: {exc}")

    # ------------------------------------------------------------------ files
    def anchor_file(self, public_id: str, digest_hex: str) -> ChainResult:
        def go():
            c = self.contracts["FileVerificationContract"]
            return self._send(c.functions.anchorFile(
                _bytes32(public_id, name="public_id"), _bytes32(digest_hex, name="digest")),
                "FileAnchored", c)
        return self._guard(go)

    def record_file_verification(self, chain_file_id: int, is_match: bool) -> ChainResult:
        c = self.contracts["FileVerificationContract"]
        res = self._send(c.functions.recordVerification(int(chain_file_id), bool(is_match)))
        res.match = bool(is_match)
        return res

    def get_file_anchor(self, chain_file_id: int) -> dict | None:
        """Read-only. Returns None when unknown / chain unreachable."""
        try:
            pub, digest, at, ver, mis = self.contracts["FileVerificationContract"] \
                .functions.getAnchor(int(chain_file_id)).call()
        except Exception:
            return None
        return {"public_id": pub.hex(), "digest": digest.hex(), "anchored_at": at,
                "verifications": ver, "mismatches": mis}

    # --------------------------------------------------------------- consents
    def grant_consent(self, did: str, recipient: str, data_type: str,
                      purpose_hash: str) -> ChainResult:
        def go():
            from web3 import Web3
            c = self.contracts["ConsentContract"]
            return self._send(c.functions.grantConsent(
                address_from_did(did), Web3.to_checksum_address(recipient), data_type,
                _bytes32(purpose_hash, name="purpose_hash")), "ConsentGranted", c)
        return self._guard(go)

    def revoke_consent(self, chain_consent_id: int) -> ChainResult:
        c = self.contracts["ConsentContract"]
        return self._send(c.functions.revokeConsent(int(chain_consent_id)))

    def has_consent(self, did: str, recipient: str, data_type: str) -> bool | None:
        """Read-only chain view; None if it cannot be determined."""
        try:
            from web3 import Web3
            return bool(self.contracts["ConsentContract"].functions.hasConsent(
                address_from_did(did), Web3.to_checksum_address(recipient), data_type).call())
        except Exception:
            return None

    # ---------------------------------------------------------------- erasure
    def record_erasure(self, erasure_ref: str) -> ChainResult:
        def go():
            c = self.contracts["ErasureLogContract"]
            return self._send(c.functions.recordErasure(_bytes32(erasure_ref, name="erasure_ref")))
        return self._guard(go)
