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
#   make docker-up    Spin up Ganache + IPFS + web (needs .env; see .env.example)
#   make docker-down  Tear down the stack
#   make clean        Remove compiled Python files and cached data

PYTHON   ?= python3
PIP      ?= pip3
PYTEST   ?= $(PYTHON) -m pytest

COMPOSE ?= docker compose

# ── Environment setup ─────────────────────────────────────────────────────────

.PHONY: setup
setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt

# ── Run the Flask application ─────────────────────────────────────────────────

.PHONY: run
run:
	EHR_DEV_MODE=$${EHR_DEV_MODE:-1} $(PYTHON) run.py

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
	$(PYTEST) tests -q

# ── Benchmark ─────────────────────────────────────────────────────────────────

.PHONY: benchmark
benchmark:
	$(PYTHON) -m app.benchmark

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
	$(COMPOSE) up --build -d

.PHONY: docker-down
docker-down:
	$(COMPOSE) down

.PHONY: docker-logs
docker-logs:
	$(COMPOSE) logs -f

# ── Clean ─────────────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -f data/benchmark_results.json data/ablation_results.json
	rm -rf data/plots/
	@echo "Cleaned."
