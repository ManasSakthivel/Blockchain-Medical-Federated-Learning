from flask import Blueprint, jsonify
import json

benchmark_bp = Blueprint('benchmark', __name__)

@benchmark_bp.route('/benchmark/results', methods=['GET'])
def get_benchmark_results():
    try:
        with open('app/benchmark_results.json', 'r') as f:
            results = json.load(f)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
