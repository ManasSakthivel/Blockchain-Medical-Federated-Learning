import logging

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFError, CSRFProtect
from werkzeug.exceptions import HTTPException

# Load environment variables from .env before anything else
load_dotenv()

db = SQLAlchemy()  # Single source of db
login_manager = LoginManager()
migrate = Migrate()
csrf = CSRFProtect()
# F-04: rate limiter — keyed by remote IP.  Storage and enabled flag come from
# config (RATELIMIT_STORAGE_URI, RATELIMIT_ENABLED); see app/config.py.
limiter = Limiter(key_func=get_remote_address, default_limits=[])

log = logging.getLogger(__name__)


def create_app(test_config: dict | None = None):
    from app.config import build_config

    app = Flask(__name__)
    app.config.update(build_config(test_config))

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)
    # Use db_migrations/ to avoid collision with Truffle's migrations/ folder
    migrate.init_app(app, db, directory="db_migrations")

    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.session_protection = 'strong'

    @login_manager.user_loader
    def load_user(session_id):
        """Session ids are ``<user id>:<session token>``; a stale token (after
        logout / password change / erasure) yields an anonymous user."""
        from app.models import User
        try:
            uid, token = str(session_id).split(':', 1)
            user = db.session.get(User, int(uid))
        except (ValueError, TypeError):
            return None
        if user is None or not user.active or user.session_token != token:
            return None
        return user

    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.admin import admin_bp
    from app.routes.doctor import doctor_bp
    from app.routes.patient import patient_bp
    from app.routes.lab import lab_bp
    from app.routes.file_verification import file_verification_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(admin_bp, url_prefix='/admin')
    app.register_blueprint(doctor_bp, url_prefix='/doctor')
    app.register_blueprint(patient_bp, url_prefix='/patient')
    app.register_blueprint(lab_bp, url_prefix='/lab')
    app.register_blueprint(file_verification_bp, url_prefix='/file-verification')

    # Schema creation at startup is a development convenience only;
    # production deployments use ``flask db upgrade`` (db_migrations/).
    if app.config.get('TESTING') or app.config.get('EHR_DEV_MODE'):
        with app.app_context():
            db.create_all()

    @app.template_filter('datetime')
    def datetime_filter(timestamp):
        """Convert Unix timestamp to readable datetime"""
        if timestamp:
            from datetime import datetime
            try:
                return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                return str(timestamp)
        return 'N/A'

    _register_security_headers(app)
    _register_error_handlers(app)
    return app


_CSP = "; ".join([
    "default-src 'self'",
    # Inline <script> blocks and on* handlers still exist in the templates, so
    # 'unsafe-inline' is required for scripts today (documented limitation in
    # docs/SECURITY_AND_PRIVACY.md). Remote scripts are limited to the pinned,
    # SRI-protected CDN origin below.
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com",
    "font-src 'self' https://cdnjs.cloudflare.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def _register_security_headers(app):
    @app.after_request
    def add_headers(resp):
        resp.headers.setdefault('Content-Security-Policy', _CSP)
        resp.headers.setdefault('X-Content-Type-Options', 'nosniff')
        resp.headers.setdefault('X-Frame-Options', 'DENY')
        resp.headers.setdefault('Referrer-Policy', 'same-origin')
        resp.headers.setdefault('Permissions-Policy', 'geolocation=(), camera=(), microphone=()')
        if request.endpoint != 'static':
            # Pages and API responses can contain PHI: never cache them.
            resp.headers.setdefault('Cache-Control', 'no-store')
        if app.config.get('SESSION_COOKIE_SECURE'):
            resp.headers.setdefault('Strict-Transport-Security', 'max-age=31536000')
        return resp


def _wants_json() -> bool:
    return (request.is_json or request.path.startswith(('/lab/api', '/file-verification/api'))
            or 'application/json' in request.headers.get('Accept', ''))


def _error_response(status: int, message: str):
    if _wants_json():
        return jsonify({'status': 'error', 'error': message}), status
    try:
        return render_template('error.html', status=status, message=message), status
    except Exception:  # pragma: no cover - template failure must not mask the error
        return message, status


def _register_error_handlers(app):
    @app.errorhandler(CSRFError)
    def handle_csrf(err):
        return _error_response(400, 'Your form expired or is invalid. Reload the page and try again.')

    @app.errorhandler(HTTPException)
    def handle_http(err):
        if err.code and 300 <= err.code < 400:  # e.g. strict-slash redirects
            return err
        return _error_response(err.code or 500, err.description or err.name)

    @app.errorhandler(Exception)
    def handle_unexpected(err):
        log.exception('Unhandled error on %s %s', request.method, request.path)
        try:
            db.session.rollback()
        except Exception:  # pragma: no cover
            log.debug('rollback after unhandled error failed', exc_info=True)
        return _error_response(500, 'Internal server error.')
