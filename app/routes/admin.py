"""Administrator pages.

Deliberately limited: an administrator can see aggregate counts and manage
account *status*. Administrators have no access to patient clinical data
(records, prescriptions, lab reports, files) through this application.
"""
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_user

from app.access import password_problem, roles_required
from app.models import Consultation, Doctor, MedicalRecord, Patient, User, db

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/dashboard')
@roles_required('admin')
def dashboard():
    return render_template(
        'admin/dashboard.html',
        total_users=User.query.count(),
        total_patients=Patient.query.count(),
        total_doctors=Doctor.query.count(),
        total_records=MedicalRecord.query.count(),
        total_consultations=Consultation.query.count(),
        recent_users=User.query.order_by(User.created_at.desc()).limit(5).all(),
    )


@admin_bp.route('/users')
@roles_required('admin')
def users():
    return render_template(
        'admin/users.html',
        users=User.query.order_by(User.created_at.desc()).limit(500).all(),
    )


@admin_bp.route('/user/<int:user_id>/deactivate', methods=['POST'])
@roles_required('admin')
def deactivate_user(user_id):
    user = db.session.get(User, user_id)
    if user is None:
        flash('User not found.', 'error')
        return redirect(url_for('admin.users'))
    if user.id == current_user.id:
        flash('You cannot deactivate your own account.', 'error')
        return redirect(url_for('admin.users'))
    user.active = False
    user.rotate_session_token()  # kills any live session immediately
    db.session.commit()
    flash('User deactivated.', 'success')
    return redirect(url_for('admin.users'))


@admin_bp.route('/reports')
@roles_required('admin')
def reports():
    """Aggregate counts only."""
    week_ago = datetime.utcnow() - timedelta(days=7)
    month_ago = datetime.utcnow() - timedelta(days=30)
    return render_template(
        'admin/reports.html',
        weekly_records=MedicalRecord.query.filter(MedicalRecord.created_at >= week_ago).count(),
        monthly_records=MedicalRecord.query.filter(MedicalRecord.created_at >= month_ago).count(),
        weekly_users=User.query.filter(User.created_at >= week_ago).count(),
        monthly_users=User.query.filter(User.created_at >= month_ago).count(),
    )


@admin_bp.route('/change-password', methods=['GET', 'POST'])
@roles_required('admin')
def change_password():
    """F-02: admin users can rotate their own password via a plain HTML form.

    Admin accounts have no profile page; this is a standalone route.
    On success the session token is rotated (all other sessions invalidated)
    and the current session is kept alive via ``login_user``.
    """
    if request.method == 'POST':
        current_password = request.form.get('current_password') or ''
        new_password = request.form.get('new_password') or ''
        confirm = request.form.get('confirm_password') or ''

        if not current_user.check_password(current_password):
            flash('Current password is incorrect.', 'error')
            return redirect(url_for('admin.change_password'))
        if new_password != confirm:
            flash('New passwords do not match.', 'error')
            return redirect(url_for('admin.change_password'))
        problem = password_problem(new_password)
        if problem:
            flash(problem, 'error')
            return redirect(url_for('admin.change_password'))

        current_user.set_password(new_password)  # also rotates session token
        db.session.commit()
        login_user(current_user)  # keep THIS session alive; all others are now invalid
        flash('Password changed successfully.', 'success')
        return redirect(url_for('admin.dashboard'))

    return render_template('admin/change_password.html')
