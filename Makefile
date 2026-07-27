# Blockchain Medical FL — Developer Makefile
# ============================================
# Usage:
#   make setup        Install Python dependencies
#   make run          Start the Flask development server
#   make test         Run the pytest test suite
#   make benchmark    Run the full FL benchmark (writes data/benchmark_results.json)
#   make ablation     Run the 4-condition ablation study
#   make plot         Generate convergence / epsilon / overhead charts
#   make db-upgrade   Apply pending database migrations
#   make docker-up    Spin up Ganache + IPFS + web (Podman preferred, Docker fallback)
#   make docker-down  Tear down the stack
#   make clean        Remove compiled Python files and cached data

PYTHON   ?= python3
PIP      ?= pip3
PYTEST   ?= $(PYTHON) -m pytest

# ── Detect compose command ────────────────────────────────────────────────────
# Tries (in order): podman-compose binary, python3 -m podman_compose, docker-compose
COMPOSE := $(shell \
    command -v podman-compose 2>/dev/null \
    || (python3 -c "import podman_compose" 2>/dev/null && echo "python3 -m podman_compose") \
    || command -v docker-compose 2>/dev/null \
    || echo "podman-compose")

# ── Environment setup ─────────────────────────────────────────────────────────

.PHONY: setup
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest
	# podman-compose may already be installed; install if missing
	python3 -c "import podman_compose" 2>/dev/null || $(PIP) install podman-compose

# ── Run the Flask application ─────────────────────────────────────────────────

.PHONY: run
run:
	$(PYTHON) run.py

# ── Database migrations ───────────────────────────────────────────────────────

.PHONY: db-upgrade
db-upgrade:
	FLASK_APP=run.py $(PYTHON) -m flask db upgrade --directory db_migrations

.PHONY: db-migrate
db-migrate:
	FLASK_APP=run.py $(PYTHON) -m flask db migrate --directory db_migrations -m "$(MSG)"

# ── Tests ─────────────────────────────────────────────────────────────────────

.PHONY: test
test:
	$(PYTEST) tests/test_differential_privacy.py \
	          tests/test_federated_simulation.py \
	          tests/test_app_models_routes.py \
	          -v --tb=short

# ── Benchmark ─────────────────────────────────────────────────────────────────

.PHONY: benchmark
benchmark:
	$(PYTHON) app/benchmark.py

.PHONY: ablation
ablation:
	$(PYTHON) scripts/ablation_study.py

# ── Charts ────────────────────────────────────────────────────────────────────

.PHONY: plot
plot:
	$(PYTHON) scripts/plot_convergence.py

# ── Podman / Docker compose ───────────────────────────────────────────────────

.PHONY: docker-up
docker-up:
	@echo "Using compose: $(COMPOSE)"
	@# Start Podman machine on macOS if needed
	@podman machine start 2>/dev/null || true
	python3 -m podman_compose up --build -d
	@echo ""
	@echo "  Ganache  → http://localhost:7545"
	@echo "  IPFS     → http://localhost:5001"
	@echo "  Web app  → http://localhost:5000"

.PHONY: docker-down
docker-down:
	python3 -m podman_compose down

.PHONY: docker-logs
docker-logs:
	python3 -m podman_compose logs -f

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f data/benchmark_results.json data/ablation_results.json
	rm -rf data/plots/
	@echo "Cleaned."
