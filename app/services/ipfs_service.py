"""
ipfs_service.py
IPFS interaction layer for the EHR system.

IMPORTANT — IPFS_STRICT_MODE:
  When True  (production / demo):  any IPFS failure raises an exception.
                                    No mock hashes are ever returned.
  When False (development only):   a clearly-labelled mock hash is returned
                                    and a prominent warning is printed so the
                                    developer knows the record is NOT backed by
                                    real IPFS content.

Set  IPFS_STRICT_MODE=true  in your .env (or shell) before any demo / thesis
evaluation to ensure the security guarantee holds.
"""

import os
import json
import requests
from flask import current_app


class IPFSUnavailableError(RuntimeError):
    """Raised when IPFS is unavailable and strict mode is enabled."""
    pass


class IPFSService:
    # ── Strict mode: read from environment, default OFF so dev still works ──
    STRICT_MODE: bool = os.environ.get("IPFS_STRICT_MODE", "false").lower() == "true"

    def __init__(self, ipfs_url: str | None = None):
        self.ipfs_url = ipfs_url or os.environ.get("IPFS_URL", "http://127.0.0.1:5001")

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _mock_hash(self, label: str = "") -> str:
        """
        Return a clearly-labelled mock hash, but ONLY in dev mode.
        Prints a very visible warning so developers cannot miss it.
        """
        import time
        mock = f"QmMOCK_DEV_ONLY_{label}_{int(time.time())}"
        print("\n" + "!" * 70)
        print("⚠️  IPFS UNAVAILABLE — MOCK HASH IN USE (dev mode only)")
        print(f"    Mock hash: {mock}")
        print("    This record is NOT backed by real IPFS content.")
        print("    Set IPFS_STRICT_MODE=true to prevent this in demos.")
        print("!" * 70 + "\n")
        return mock

    def _handle_ipfs_error(self, context: str, error: Exception) -> str:
        """
        Central error handler.
        In strict mode → raise IPFSUnavailableError (bubbles up to caller).
        In dev mode    → return a clearly-labelled mock hash.
        """
        msg = f"IPFS error during '{context}': {error}"
        print(f"❌ {msg}")
        if self.STRICT_MODE:
            raise IPFSUnavailableError(msg) from error
        return self._mock_hash(context)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def upload_file(self, file_path: str) -> str:
        """
        Upload a local file to IPFS.
        Returns the IPFS CID (hash) string.
        Raises IPFSUnavailableError in strict mode if IPFS is unreachable.
        """
        try:
            with open(file_path, 'rb') as fh:
                response = requests.post(
                    f"{self.ipfs_url}/api/v0/add",
                    files={'file': fh},
                    timeout=30
                )
            if response.status_code == 200:
                cid = response.json()['Hash']
                print(f"✅ IPFS upload success: {cid}")
                return cid
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except IPFSUnavailableError:
            raise  # already formatted
        except Exception as e:
            return self._handle_ipfs_error("upload_file", e)

    def upload_json(self, data: dict) -> str | None:
        """
        Upload a JSON-serialisable dict to IPFS as a file.
        Returns the CID string, or None on failure (strict mode raises).
        """
        try:
            json_bytes = json.dumps(data).encode()
            response = requests.post(
                f"{self.ipfs_url}/api/v0/add",
                files={'file': ('data.json', json_bytes, 'application/json')},
                timeout=30
            )
            if response.status_code == 200:
                cid = response.json()['Hash']
                print(f"✅ IPFS JSON upload success: {cid}")
                return cid
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except IPFSUnavailableError:
            raise
        except Exception as e:
            if self.STRICT_MODE:
                raise IPFSUnavailableError(f"IPFS JSON upload failed: {e}") from e
            print(f"❌ IPFS JSON upload error: {e}")
            return None

    def get_file(self, ipfs_hash: str) -> bytes | None:
        """
        Retrieve raw file bytes from IPFS by CID.
        Returns None on failure (strict mode raises).
        """
        # Never attempt to fetch a mock hash
        if ipfs_hash.startswith("QmMOCK_DEV_ONLY_"):
            print(f"⚠️  Skipping fetch for mock IPFS hash: {ipfs_hash}")
            return None
        try:
            response = requests.post(
                f"{self.ipfs_url}/api/v0/cat",
                params={'arg': ipfs_hash},
                timeout=30
            )
            if response.status_code == 200:
                return response.content
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except IPFSUnavailableError:
            raise
        except Exception as e:
            if self.STRICT_MODE:
                raise IPFSUnavailableError(f"IPFS get failed: {e}") from e
            print(f"❌ IPFS get_file error: {e}")
            return None

    def get_json(self, ipfs_hash: str) -> dict | None:
        """Retrieve and decode JSON content from IPFS."""
        content = self.get_file(ipfs_hash)
        if content:
            try:
                return json.loads(content.decode('utf-8'))
            except Exception as e:
                print(f"❌ JSON decode error for {ipfs_hash}: {e}")
        return None

    def pin_file(self, ipfs_hash: str) -> bool:
        """
        Pin a file on IPFS to prevent garbage collection.
        Returns True on success, False on failure (strict mode raises for non-mock hashes).
        """
        # Mock hashes don't need pinning; just acknowledge
        if ipfs_hash.startswith("QmMOCK_DEV_ONLY_"):
            print(f"ℹ️  Skipping pin for mock hash (dev mode): {ipfs_hash}")
            return True
        try:
            response = requests.post(
                f"{self.ipfs_url}/api/v0/pin/add",
                params={'arg': ipfs_hash},
                timeout=30
            )
            if response.status_code == 200:
                print(f"✅ IPFS pin success: {ipfs_hash}")
                return True
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
        except IPFSUnavailableError:
            raise
        except Exception as e:
            if self.STRICT_MODE:
                raise IPFSUnavailableError(f"IPFS pin failed: {e}") from e
            print(f"❌ IPFS pin error: {e}")
            return False

    def is_mock_hash(self, ipfs_hash: str) -> bool:
        """Returns True if this hash is a dev-mode placeholder (not real IPFS content)."""
        return ipfs_hash.startswith("QmMOCK_DEV_ONLY_")

    def health_check(self) -> dict:
        """Check IPFS daemon connectivity. Returns status dict."""
        try:
            response = requests.post(
                f"{self.ipfs_url}/api/v0/id",
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    "connected": True,
                    "peer_id": data.get("ID"),
                    "addresses": data.get("Addresses", []),
                    "strict_mode": self.STRICT_MODE,
                }
            return {"connected": False, "error": response.text, "strict_mode": self.STRICT_MODE}
        except Exception as e:
            return {"connected": False, "error": str(e), "strict_mode": self.STRICT_MODE}