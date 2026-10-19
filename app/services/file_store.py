"""Encrypted file storage and authenticated retrieval.

Files are never written in plaintext and never placed under ``static/``.
Authorisation is decided by the caller from the ``StoredFile`` row.
"""
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import current_app

from app import db
from app.models import StoredFile
from app.services.crypto_service import CryptoService

# extension -> accepted leading bytes (content sniffing, not just the name)
_MAGIC = {
    "pdf": (b"%PDF-",),
    "png": (b"\x89PNG\r\n\x1a\n",),
    "jpg": (b"\xff\xd8\xff",),
    "jpeg": (b"\xff\xd8\xff",),
    "gif": (b"GIF87a", b"GIF89a"),
}
_TEXT_EXT = {"txt", "csv"}
ALLOWED_EXTENSIONS = set(_MAGIC) | _TEXT_EXT
IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_BYTES = 16 * 1024 * 1024
_MIME = {"pdf": "application/pdf", "png": "image/png", "jpg": "image/jpeg",
         "jpeg": "image/jpeg", "gif": "image/gif", "txt": "text/plain",
         "csv": "text/csv"}


class UploadRejected(ValueError):
    """User-facing validation failure."""


def get_crypto() -> CryptoService:
    svc = current_app.extensions.get("ehr_crypto")
    if svc is None:
        svc = CryptoService.from_env(
            allow_ephemeral=bool(current_app.config.get("TESTING")
                                 or current_app.config.get("EHR_DEV_MODE")))
        current_app.extensions["ehr_crypto"] = svc
    return svc


def upload_dir() -> Path:
    path = Path(current_app.config["UPLOAD_DIR"])
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return path


def validate_upload(filename: str, data: bytes, allowed=ALLOWED_EXTENSIONS) -> str:
    """Return the normalised extension or raise UploadRejected."""
    if not filename or "." not in filename:
        raise UploadRejected("File must have an extension.")
    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in allowed:
        raise UploadRejected(f"File type '.{ext}' is not allowed.")
    if not data:
        raise UploadRejected("File is empty.")
    if len(data) > MAX_BYTES:
        raise UploadRejected("File is too large.")
    if ext in _MAGIC and not any(data.startswith(m) for m in _MAGIC[ext]):
        raise UploadRejected("File content does not match its extension.")
    if ext in _TEXT_EXT:
        try:
            data[:4096].decode("utf-8")
        except UnicodeDecodeError:
            raise UploadRejected("Text file must be UTF-8.")
    return ext


def _aad(public_id: str, patient_id: int) -> bytes:
    return f"{public_id}:{patient_id}".encode()


def store_file(*, patient_id: int, uploader_user_id: int | None, filename: str,
               data: bytes, category: str = "medical",
               allowed=ALLOWED_EXTENSIONS) -> StoredFile:
    """Encrypt ``data`` and persist it. Adds the row to the session (no commit)."""
    ext = validate_upload(filename, data, allowed)
    crypto = get_crypto()
    public_id = uuid.uuid4().hex
    ciphertext, nonce, wrapped = crypto.encrypt(data, _aad(public_id, patient_id))

    rel = f"{public_id}.enc"
    path = upload_dir() / rel
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(ciphertext)

    safe_name = os.path.basename(filename.replace("\\", "/"))[:255] or f"file.{ext}"
    row = StoredFile(
        public_id=public_id, patient_id=patient_id, uploaded_by_user_id=uploader_user_id,
        category=category, original_filename=safe_name, extension=ext, size_bytes=len(data),
        anchor_digest=crypto.anchor_digest(data), nonce=nonce, wrapped_dek=wrapped,
        kek_id=crypto.kek_id, local_path=rel)
    db.session.add(row)
    return row


def read_plaintext(stored: StoredFile) -> bytes:
    if stored.is_erased or not stored.wrapped_dek or not stored.local_path:
        raise FileNotFoundError("file has been erased")
    ciphertext = (upload_dir() / stored.local_path).read_bytes()
    return get_crypto().decrypt(ciphertext, stored.nonce, stored.wrapped_dek,
                                _aad(stored.public_id, stored.patient_id))


def mime_for(stored: StoredFile) -> str:
    return _MIME.get(stored.extension, "application/octet-stream")


def shred(stored: StoredFile) -> None:
    """Crypto-shred: destroy the wrapped key, delete local ciphertext."""
    if stored.local_path:
        try:
            (upload_dir() / stored.local_path).unlink()
        except FileNotFoundError:
            pass
    stored.wrapped_dek = None
    stored.local_path = None
    stored.erased_at = datetime.utcnow()
