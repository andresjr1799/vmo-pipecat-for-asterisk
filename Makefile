.PHONY: up down logs test lint e2e e2e-modular e2e-full-agent e2e-transfer build help

## — Development ——————————————————————————————————————————————
up: ## Start all containers (vmo_pipecat + ast-1 + ast-2)
	docker compose up --build -d
	@echo "Waiting for healthchecks..."
	@docker compose ps

down: ## Stop and remove containers
	docker compose down --remove-orphans

logs: ## Tail logs (all services)
	docker compose logs -f

logs-pipecat: ## Tail vmo_pipecat logs only
	docker compose logs -f vmo_pipecat

build: ## Build Docker images without starting
	docker compose build

## — Testing ——————————————————————————————————————————————————
test: ## Run unit + integration tests (local, no Docker)
	pytest tests/ -q --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=vmo_pipecat --cov-report=term-missing -q

lint: ## Run ruff linter
	ruff check vmo_pipecat/ tests/

## — E2E (requires running containers) ————————————————————————
e2e-modular: ## E2E test: modular pipeline (mock providers)
	pytest tests/integration/test_e2e_modular_deepgram_openai_eleven.py -v

e2e-full-agent: ## E2E test: full-agent Deepgram Voice Agent
	pytest tests/integration/test_e2e_full_agent_deepgram.py -v

e2e-transfer: ## E2E test: transfer_call tool flow
	pytest tests/integration/test_e2e_transfer.py -v

## — Help —————————————————————————————————————————————————————
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
