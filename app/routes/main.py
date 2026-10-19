from flask import Blueprint, current_app, redirect, render_template, request, url_for
from flask_login import current_user

from app.federated_sim_engine import FederatedSimulation

main_bp = Blueprint('main', __name__)


def _int_arg(name, default=None, lo=None, hi=None):
    """Parse an integer query argument; fall back to ``default`` when absent/invalid."""
    raw = request.args.get(name, '')
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


@main_bp.route('/federated-sim')
def federated_simulation():
    """Public demo of the in-process simulation (Cleveland data, 3 nodes, default
    DP noise). It exposes no user data. The round count is capped (``MAX_SIM_ROUNDS``)
    so the endpoint cannot be used to burn CPU."""
    max_rounds = current_app.config.get('MAX_SIM_ROUNDS', 20)
    n_rounds = _int_arg('n_rounds', default=3, lo=1, hi=max_rounds)
    tamper_round = _int_arg('tamper_round', lo=1, hi=n_rounds)
    tamper_node = _int_arg('tamper_node', lo=1, hi=3)
    sim = FederatedSimulation(n_nodes=3, n_rounds=n_rounds)
    round_logs = sim.run_simulation(
        tamper_round=tamper_round - 1 if tamper_round else None,
        tamper_node=tamper_node - 1 if tamper_node else None,
    )
    return render_template('federated_simulation.html', round_logs=round_logs,
                           n_rounds=n_rounds, tamper_round=tamper_round or '',
                           tamper_node=tamper_node or '')


@main_bp.route('/')
def home():
    return render_template('main/home.html')


@main_bp.route('/dashboard')
def dashboard():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    target = {'admin': 'admin.dashboard', 'doctor': 'doctor.dashboard',
              'patient': 'patient.dashboard', 'lab': 'lab.dashboard'}.get(current_user.role)
    return redirect(url_for(target or 'main.home'))
