"""Application configuration.

Secrets are never given usable defaults: outside of testing / explicit
development mode the app refuses to start without ``SECRET_KEY``.
"""
import logging
import os
import secrets
import tempfile

log = logging.getLogger(__name__)

_TRUE = {"1", "true", "yes", "on"}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in _TRUE


class ConfigError(RuntimeError):
    """Raised when the configuration is unsafe or incomplete."""


def build_config(overrides: dict | None = None) -> dict:
    """Return the Flask config dict. ``overrides`` (tests) win over the env."""
    overrides = overrides or {}
    testing = bool(overrides.get("TESTING"))
    dev = env_flag("EHR_DEV_MODE")  # explicit opt-in for local development

    secret = overrides.get("SECRET_KEY") or os.environ.get("SECRET_KEY")
    if not secret:
        if testing or dev:
            secret = secrets.token_hex(32)  # ephemeral; sessions reset on restart
        else:
            raise ConfigError(
                "SECRET_KEY is not set. Generate one with "
                "`python -c 'import secrets; print(secrets.token_hex(32))'` "
                "and export it (or set EHR_DEV_MODE=1 for local development)."
            )
    if secret in {"your-secret-key-here", "dev-secret-change-in-production",
                  "change-this-to-a-long-random-string-before-any-demo"}:
        raise ConfigError("SECRET_KEY is set to a known placeholder value.")

    upload_dir = os.environ.get("UPLOAD_DIR") or os.path.join(
        tempfile.gettempdir() if (testing or dev) else "/var/lib/ehr", "ehr_uploads"
    )

    cfg = {
        "SECRET_KEY": secret,
        "SQLALCHEMY_DATABASE_URI": os.environ.get("DATABASE_URL", "sqlite:///ehr.db"),
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SESSION_COOKIE_HTTPONLY": True,
        "SESSION_COOKIE_SAMESITE": "Lax",
        "SESSION_COOKIE_SECURE": env_flag("SESSION_COOKIE_SECURE", default=not (testing or dev)),
        "REMEMBER_COOKIE_HTTPONLY": True,
        "REMEMBER_COOKIE_SAMESITE": "Lax",
        "REMEMBER_COOKIE_SECURE": env_flag("SESSION_COOKIE_SECURE", default=not (testing or dev)),
        "MAX_CONTENT_LENGTH": 16 * 1024 * 1024,
        "WTF_CSRF_TIME_LIMIT": None,  # token valid for the session lifetime
        "UPLOAD_DIR": upload_dir,
        "EHR_DEV_MODE": dev,
        "MAX_SIM_ROUNDS": 20,
        # F-04: Login rate limiting via Flask-Limiter.
        # Disabled in TESTING to avoid spurious 429 errors in the test suite.
        # In production, set RATELIMIT_STORAGE_URI (e.g. "redis://localhost:6379/0")
        # to share limits across gunicorn workers. Without it, limits are per-process
        # and do not survive restarts — documented limitation for single-worker deploys.
        "RATELIMIT_ENABLED": not testing,
        "RATELIMIT_STORAGE_URI": os.environ.get("RATELIMIT_STORAGE_URI", "memory://"),
        "RATELIMIT_HEADERS_ENABLED": True,
        "RATELIMIT_DEFAULT_LIMITS_DEDUCT_WHEN_FAILED": True,
    }
    cfg.update(overrides)

    # Warn operators (but do not fail) when login throttling is per-process only.
    # Skipped in testing and dev mode where memory:// is the expected default.
    if (cfg.get("RATELIMIT_ENABLED")
            and not (testing or dev)
            and str(cfg.get("RATELIMIT_STORAGE_URI", "memory://")).startswith("memory://")):
        log.warning(
            "RATELIMIT_STORAGE_URI is not set or uses memory://: login rate limits "
            "are per-process, reset on restart and are NOT shared across gunicorn "
            "workers. Set RATELIMIT_STORAGE_URI to a shared store "
            "(e.g. redis://host:6379/0) in production."
        )

    return cfg
