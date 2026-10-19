from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy.exc import IntegrityError
from werkzeug.security import check_password_hash, generate_password_hash

from app import limiter
from app.access import password_problem, valid_email
from app.models import Doctor, Patient, User, db

auth_bp = Blueprint('auth', __name__)

# Roles that may be self-registered. 'admin' and 'lab' accounts are created
# only through the CLI (``flask create-user``), never through the web form.
SELF_REGISTER_ROLES = ('patient', 'doctor')

# Used to equalise timing when the e-mail is unknown (user enumeration, A-13).
_DUMMY_HASH = generate_password_hash('not-a-real-password')


def _form(name, default=''):
    return (request.form.get(name) or default).strip()


def _first_nonempty(name):
    """The register form has one `phone`/`address` input per role section."""
    for value in request.form.getlist(name):
        if value.strip():
            return value.strip()
    return ''


def _safe_next(target):
    """Only allow same-site relative redirects."""
    if target and target.startswith('/') and not target.startswith('//') and '\\' not in target:
        return target
    return None


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit(
    "10 per minute",
    error_message="Too many login attempts. Please wait before trying again.",
    methods=["POST"],
)
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        email = _form('email').lower()
        password = request.form.get('password') or ''

        user = User.query.filter_by(email=email).first() if email else None
        if user is not None:
            ok = user.check_password(password)
        else:
            check_password_hash(_DUMMY_HASH, password)
            ok = False

        if ok and user.active:
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(_safe_next(request.args.get('next')) or url_for('main.home'))
        flash('Invalid email or password.', 'error')

    return render_template('auth/login.html')


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.home'))

    if request.method == 'POST':
        username = _form('username')
        email = _form('email').lower()
        password = request.form.get('password') or ''
        role = _form('role')

        def fail(message):
            flash(message, 'error')
            return redirect(url_for('auth.register'))

        if role not in SELF_REGISTER_ROLES:
            return fail('Invalid role.')
        if not username or len(username) > 80:
            return fail('Username is required (max 80 characters).')
        if not valid_email(email):
            return fail('A valid e-mail address is required.')
        problem = password_problem(password)
        if problem:
            return fail(problem)
        if request.form.get('confirm_password') not in (None, password):
            return fail('Passwords do not match.')
        if not _form('first_name') or not _form('last_name'):
            return fail('First and last name are required.')

        # F-04: uniform message for all duplicate-identity conflicts to avoid
        # leaking whether a specific e-mail or username is in use.
        if User.query.filter_by(email=email).first() or \
                User.query.filter_by(username=username).first():
            return fail('Registration failed. Check your input and try again.')

        # Validate role-specific input BEFORE touching the database so that a
        # bad form never leaves an orphan user behind (A-07).
        profile = None
        if role == 'patient':
            dob_raw = _form('date_of_birth')
            dob = None
            if dob_raw:
                try:
                    dob = datetime.strptime(dob_raw, '%Y-%m-%d').date()
                except ValueError:
                    return fail('Invalid date of birth. Use YYYY-MM-DD.')
                if dob > datetime.utcnow().date():
                    return fail('Date of birth cannot be in the future.')
        else:
            license_number = _form('license_number')
            if not license_number or not _form('specialization'):
                return fail('Specialization and license number are required.')
            try:
                experience = int(_form('experience_years', '0') or 0)
            except ValueError:
                return fail('Experience must be a whole number.')
            if not 0 <= experience <= 70:
                return fail('Experience must be between 0 and 70 years.')

        # DID: public address of a fresh secp256k1 key; the private key is
        # discarded and never stored.
        from eth_account import Account
        acct = Account.create()
        user = User(username=username, email=email, role=role,
                    did=f"did:ethr:{acct.address}")
        user.set_password(password)

        try:
            db.session.add(user)
            db.session.flush()  # assigns user.id inside the transaction
            if role == 'patient':
                profile = Patient(
                    user_id=user.id,
                    first_name=_form('first_name'),
                    last_name=_form('last_name'),
                    date_of_birth=dob,
                    gender=_form('gender', 'unspecified'),
                    phone=_first_nonempty('phone') or '-',
                    address=_first_nonempty('address') or '-',
                    emergency_contact=_form('emergency_contact', '-') or '-',
                )
            else:
                profile = Doctor(
                    user_id=user.id,
                    first_name=_form('first_name'),
                    last_name=_form('last_name'),
                    specialization=_form('specialization'),
                    license_number=license_number,
                    phone=_first_nonempty('phone') or '-',
                    address=_first_nonempty('address') or '-',
                    experience_years=experience,
                    education=_form('education', '-') or '-',
                )
            db.session.add(profile)
            db.session.commit()  # user + profile are committed atomically
        except IntegrityError:
            db.session.rollback()
            return fail('Registration failed: username, e-mail or license number already in use.')

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('auth/register.html')


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    # Rotating the token invalidates every copy of the session cookie.
    current_user.rotate_session_token()
    db.session.commit()
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('main.home'))
