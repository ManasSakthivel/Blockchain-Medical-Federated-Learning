import numpy as np
def federated_simulation():
    # Generate data for 3 hospitals
    hospitals = [generate_hospital_data(random_state=i) for i in range(3)]
    # Run a federated learning round
    local_hashes, global_hash = federated_round(hospitals, None)  # BlockchainService not needed for display
    return render_template('federated_simulation.html', local_hashes=local_hashes, global_hash=global_hash)

from flask import Blueprint, render_template, redirect, url_for
from flask_login import current_user
import numpy as np

from app.federated_sim_engine import FederatedSimulation

main_bp = Blueprint('main', __name__)


@main_bp.route('/federated-sim')
def federated_simulation():
    # Get params from query string
    from flask import request
    n_rounds = int(request.args.get('n_rounds', 3))
    tamper_round = request.args.get('tamper_round', '')
    tamper_node = request.args.get('tamper_node', '')
    try:
        tamper_round = int(tamper_round) if tamper_round else None
    except Exception:
        tamper_round = None
    try:
        tamper_node = int(tamper_node)-1 if tamper_node else None
    except Exception:
        tamper_node = None
    sim = FederatedSimulation(n_nodes=3, n_rounds=n_rounds)
    round_logs = sim.run_simulation(tamper_round=tamper_round-1 if tamper_round else None, tamper_node=tamper_node) if n_rounds else []
    return render_template('federated_simulation.html', round_logs=round_logs, n_rounds=n_rounds, tamper_round=tamper_round, tamper_node=(tamper_node+1 if tamper_node is not None else ''))

@main_bp.route('/')
def home():
    return render_template('main/home.html')

@main_bp.route('/dashboard')
def dashboard():
    if not current_user.is_authenticated:
        return redirect(url_for('auth.login'))
    
    if current_user.role == 'admin':
        return redirect(url_for('admin.dashboard'))
    elif current_user.role == 'doctor':
        return redirect(url_for('doctor.dashboard'))
    elif current_user.role == 'patient':
        return redirect(url_for('patient.dashboard'))
    
    return redirect(url_for('main.home')) 