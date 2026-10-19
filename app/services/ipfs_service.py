"""Minimal IPFS HTTP-API client.

* Never fabricates content ids: if the node cannot be reached, an
  :class:`IPFSUnavailableError` is raised (there is no "mock hash" mode).
* Configured from ``IPFS_API_URL`` only; unset means "no IPFS".
* Callers must only ever send **ciphertext**: IPFS is a public, content-addressed
  network and pinned data cannot be recalled from other nodes.
"""
import os

import requests


class IPFSUnavailableError(RuntimeError):
    """IPFS is not configured, not reachable, or returned an error."""


class IPFSService:
    def __init__(self, url: str, session: requests.Session | None = None, timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "IPFSService":
        url = os.environ.get("IPFS_API_URL")
        if not url:
            raise IPFSUnavailableError("IPFS_API_URL is not set")
        return cls(url)

    def _post(self, path: str, **kwargs):
        try:
            resp = self.session.post(f"{self.url}/api/v0/{path}", timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            raise IPFSUnavailableError(f"IPFS {path} failed: {type(exc).__name__}") from exc

    def add_bytes(self, data: bytes, pin: bool = True) -> str:
        """Add ``data`` and return its CID."""
        resp = self._post("add", params={"pin": str(bool(pin)).lower()},
                          files={"file": ("blob", data)})
        try:
            return resp.json()["Hash"]
        except (ValueError, KeyError) as exc:
            raise IPFSUnavailableError("IPFS add returned an unexpected response") from exc

    def get_bytes(self, cid: str) -> bytes:
        return self._post("cat", params={"arg": cid}).content

    def unpin(self, cid: str) -> bool:
        """Unpin on THIS node. Does not (cannot) remove copies held elsewhere."""
        try:
            self._post("pin/rm", params={"arg": cid})
            return True
        except IPFSUnavailableError:
            return False

    def health(self) -> bool:
        try:
            self._post("version")
            return True
        except IPFSUnavailableError:
            return False
