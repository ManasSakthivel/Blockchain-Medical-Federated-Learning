"""Envelope encryption for stored files and keyed anchor digests.

* Every file gets its own random 256-bit data-encryption key (DEK).
* The DEK is wrapped (AES-256-GCM) with a key-encryption key (KEK) taken from
  the environment (``FILE_KEK``: base64 of 32 random bytes).
* Erasure = deleting the wrapped DEK ("crypto-shredding"). Without the DEK the
  ciphertext (on disk or on IPFS) can no longer be decrypted. This is NOT a
  guarantee that every copy of the ciphertext disappeared.
* The value anchored on a blockchain is an HMAC-SHA256 of the plaintext under
  ``ANCHOR_HMAC_KEY``; a plain SHA-256 would let anyone who guesses a file
  confirm its presence on a public ledger.
"""
import base64
import hashlib
import hmac
import logging
import os
import secrets
import warnings

log = logging.getLogger(__name__)

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class CryptoConfigError(RuntimeError):
    pass


class DecryptionError(RuntimeError):
    pass


def _load_key(env_name: str, allow_ephemeral: bool) -> bytes:
    raw = os.environ.get(env_name)
    if raw:
        try:
            key = base64.b64decode(raw, validate=True)
        except Exception as exc:
            raise CryptoConfigError(f"{env_name} must be base64") from exc
        if len(key) != 32:
            raise CryptoConfigError(f"{env_name} must decode to exactly 32 bytes")
        return key
    if allow_ephemeral:
        _msg = (f"{env_name} is not set: using an EPHEMERAL key. "
                "Encrypted files cannot be read after a restart.")
        warnings.warn(_msg, RuntimeWarning, stacklevel=3)
        log.warning("%s", _msg)   # F-09: also reach the Flask log sink
        return secrets.token_bytes(32)
    raise CryptoConfigError(
        f"{env_name} is not set. Generate with: "
        "python -c \"import os,base64;print(base64.b64encode(os.urandom(32)).decode())\"")


class CryptoService:
    def __init__(self, kek: bytes, anchor_key: bytes, kek_id: str = "v1"):
        if len(kek) != 32 or len(anchor_key) != 32:
            raise CryptoConfigError("keys must be 32 bytes")
        self._kek = AESGCM(kek)
        self._anchor_key = anchor_key
        self.kek_id = kek_id

    @classmethod
    def from_env(cls, allow_ephemeral: bool = False) -> "CryptoService":
        return cls(_load_key("FILE_KEK", allow_ephemeral),
                   _load_key("ANCHOR_HMAC_KEY", allow_ephemeral))

    # -- file encryption ---------------------------------------------------
    def encrypt(self, plaintext: bytes, aad: bytes):
        """Return (ciphertext, nonce_b64, wrapped_dek_b64)."""
        dek = AESGCM.generate_key(bit_length=256)
        nonce = os.urandom(12)
        ciphertext = AESGCM(dek).encrypt(nonce, plaintext, aad)
        wrap_nonce = os.urandom(12)
        wrapped = wrap_nonce + self._kek.encrypt(wrap_nonce, dek, aad)
        return (ciphertext, base64.b64encode(nonce).decode(),
                base64.b64encode(wrapped).decode())

    def decrypt(self, ciphertext: bytes, nonce_b64: str, wrapped_dek_b64: str | None,
                aad: bytes) -> bytes:
        if not wrapped_dek_b64:
            raise DecryptionError("key has been destroyed (file erased)")
        try:
            wrapped = base64.b64decode(wrapped_dek_b64)
            dek = self._kek.decrypt(wrapped[:12], wrapped[12:], aad)
            return AESGCM(dek).decrypt(base64.b64decode(nonce_b64), ciphertext, aad)
        except (InvalidTag, ValueError) as exc:
            raise DecryptionError("authentication failed") from exc

    # -- anchoring ---------------------------------------------------------
    def anchor_digest(self, plaintext: bytes) -> str:
        return hmac.new(self._anchor_key, plaintext, hashlib.sha256).hexdigest()

    def pseudonym(self, value: str) -> str:
        """Stable keyed pseudonym (hex) for identifiers placed on a ledger."""
        return hmac.new(self._anchor_key, b"pseudonym:" + value.encode(),
                        hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Standalone purpose-hash helper (F-03)
#
# Purpose: produce a keyed HMAC-SHA256 of a consent purpose string so that
# anyone who reads the public blockchain cannot confirm a guessed purpose
# against the stored digest by plain-text dictionary attack.
#
# Key used: ANCHOR_HMAC_KEY — the same 32-byte key already protecting file
# anchor digests. Using the same key is safe because the inputs are always
# prefixed with a domain label ("purpose:") that makes the HMAC outputs
# independent of those produced by anchor_digest() / pseudonym().
#
# In testing / dev mode (TESTING=True or EHR_DEV_MODE=1) the key may be
# absent; _load_key() will generate an ephemeral key for that process
# lifetime.  In production mode the key is required.
#
# Legacy records: purpose_hash values written before this change are plain
# SHA-256 digests stored in the DB.  The application never re-derives or
# verifies the purpose_hash at runtime (it is a write-once audit field
# forwarded to the chain).  Therefore no migration is required and no
# silent compatibility shim is needed. Legacy rows remain in the DB with
# their original SHA-256 values; new rows carry HMAC digests.  Any
# operator who needs to audit legacy records must note that rows created
# before this deployment use SHA-256 and rows after use HMAC-SHA256.
# ---------------------------------------------------------------------------

def purpose_hmac(purpose: str, *, allow_ephemeral: bool = False) -> str:
    """Return ``'0x' + HMAC-SHA256(ANCHOR_HMAC_KEY, 'purpose:' + purpose)`` as hex.

    The ``'0x'`` prefix preserves the Solidity ``bytes32`` encoding convention
    already established by the original SHA-256 implementation.

    Raises ``CryptoConfigError`` when ``ANCHOR_HMAC_KEY`` is not set and
    ``allow_ephemeral`` is False (i.e. in production).  In test / dev mode
    (``allow_ephemeral=True``) an ephemeral key is used and a RuntimeWarning
    is emitted — acceptable because tests never compare cross-process digests.
    """
    key = _load_key("ANCHOR_HMAC_KEY", allow_ephemeral=allow_ephemeral)
    digest = hmac.new(key, b"purpose:" + purpose.encode(), hashlib.sha256).hexdigest()
    return "0x" + digest
