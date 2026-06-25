SIM_COMPETITION := pokemon-tcg-ai-battle
STRAT_COMPETITION := pokemon-tcg-ai-battle-challenge-strategy
DATA_DIR       := data
TOKEN_FILE     := .kaggle/access_token

SIM_DATA_DIR   := data/sim_sample
RAW_DIR        := data/raw
SUBMISSION_FILE ?= submit/submission.tar.gz
SUBMISSION_MSG  ?= "agent: rule-based baseline"

.PHONY: all install download sim-download train submit test lint format clean

all: install download
	@echo ""
	@echo "========================================================"
	@echo "  Setup complete. Next steps:"
	@echo "    make sim-download    # get the simulation SDK"
	@echo "    make test            # run tests"
	@echo "    make gauntlet        # run agent tournament"
	@echo "========================================================"

install: .uv_sync
	uv pip install -e .
	@$(MAKE) _ensure_kaggle_auth
	@echo ""
	@echo "All set. Run 'make download' to get card data, then 'make sim-download' for the SDK."

download:
	@mkdir -p $(RAW_DIR); \
	$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Downloading $(STRAT_COMPETITION) card data..."; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions download \
		-c $(STRAT_COMPETITION) -f EN_Card_Data.csv -p $(RAW_DIR) && \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions download \
		-c $(STRAT_COMPETITION) -f JP_Card_Data.csv -p $(RAW_DIR) || true; \
	echo "  Card data ready in $(RAW_DIR)/"

sim-download:
	@mkdir -p $(SIM_DATA_DIR); \
	$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions download \
		-c $(SIM_COMPETITION) -p $(SIM_DATA_DIR) || { \
		exit_code=$$?; \
		echo ""; \
		echo "================================================================"; \
		echo " Download failed (403 Forbidden)."; \
		echo " You must join the Simulation competition:"; \
		echo "   https://www.kaggle.com/competitions/$(SIM_COMPETITION)"; \
		echo "================================================================"; \
		exit $$exit_code; \
	}; \
	cd $(SIM_DATA_DIR) && unzip -o *.zip && rm -f *.zip; \
	if [ -d sample_submission/cg ]; then \
		cp -r sample_submission/cg cg; \
	fi; \
	echo "  Simulation SDK ready in $(SIM_DATA_DIR)/"

gauntlet:
	@PYTHONPATH=$(SIM_DATA_DIR):$$PYTHONPATH uv run python scripts/gauntlet.py $(ARGS)

build-submit:
	@uv run python scripts/build_submission.py $(ARGS)

submit:
	@$(MAKE) _ensure_kaggle_token; \
	TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	[ ! -f $(SUBMISSION_FILE) ] && { echo "ERROR: $(SUBMISSION_FILE) not found — run 'make build-submit' first"; exit 1; }; \
	echo "Submitting $(SUBMISSION_FILE) to $(SIM_COMPETITION)..."; \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions submit \
		-c $(SIM_COMPETITION) \
		-f $(SUBMISSION_FILE) \
		-m "$(SUBMISSION_MSG)" && \
	echo "" && \
	echo "✓ Submitted! Checking leaderboard..." && \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions leaderboard \
		-c $(SIM_COMPETITION) --show

test:
	@uv run pytest tests/ -v $(ARGS)

lint:
	@uv run ruff check src/ scripts/ tests/

format:
	@uv run ruff format src/ scripts/ tests/ --check

format-fix:
	@uv run ruff format src/ scripts/ tests/

.uv_sync: pyproject.toml uv.lock
	uv sync --extra dev
	@touch .uv_sync

_ensure_kaggle_token:
	@mkdir -p .kaggle; \
	PLACEHOLDER="KGAT_your-kaggle-api-token-here"; \
	TOKEN=""; \
	\
	if [ -n "$$KAGGLE_API_TOKEN" ]; then \
		TOKEN="$$KAGGLE_API_TOKEN"; \
		if [ ! -f $(TOKEN_FILE) ] || [ "$$(cat $(TOKEN_FILE))" != "$$TOKEN" ]; then \
			printf '%s' "$$TOKEN" > $(TOKEN_FILE); \
			chmod 600 $(TOKEN_FILE); \
		fi; \
	elif [ -f $(TOKEN_FILE) ] && [ -s $(TOKEN_FILE) ] && ! grep -q "$$PLACEHOLDER" $(TOKEN_FILE) 2>/dev/null; then \
		TOKEN=$$(cat $(TOKEN_FILE)); \
	elif [ -f ~/.kaggle/kaggle.json ] && ! grep -q "your-kaggle-username" ~/.kaggle/kaggle.json 2>/dev/null; then \
		echo "Using legacy credentials from ~/.kaggle/kaggle.json"; \
		exit 0; \
	else \
		if [ -f .kaggle/access_token.example ]; then \
			cp .kaggle/access_token.example $(TOKEN_FILE); \
		else \
			printf 'KGAT_your-kaggle-api-token-here\n' > $(TOKEN_FILE); \
		fi; \
		chmod 600 $(TOKEN_FILE); \
		echo ""; \
		echo " Token template written to $(TOKEN_FILE)."; \
		echo ""; \
		echo " To configure:"; \
		echo "   1. Go to https://www.kaggle.com/settings"; \
		echo "   2. Under API, click 'Create New Token'"; \
		echo "   3. Copy the token (starts with KGAT_)"; \
		echo "   4. Paste it into $(TOKEN_FILE) (and nothing else)"; \
		echo "   5. Run 'make download' again to verify"; \
		exit 1; \
	fi

_ensure_kaggle_auth: _ensure_kaggle_token
	@TOKEN="$$(cat $(TOKEN_FILE) 2>/dev/null)"; \
	[ -z "$$TOKEN" ] && TOKEN="$$KAGGLE_API_TOKEN"; \
	echo "Verifying..." && \
	KAGGLE_API_TOKEN="$$TOKEN" uv run kaggle competitions list >/dev/null 2>&1 && \
	echo "  Authenticated successfully." || \
	{ echo "  WARNING: Authentication check failed."; exit 1; }

clean:
	rm -f .uv_sync
	rm -rf submit/ workspace/results/
