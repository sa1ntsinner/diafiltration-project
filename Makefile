# Convenience targets — everything also works as plain python commands.
PY ?= python

.PHONY: help install figures quick test bench optimum dashboard clean
help:
	@echo "make install    - editable install with dev extras"
	@echo "make figures    - regenerate every figure and results/results.json"
	@echo "make quick      - same, with a reduced Monte-Carlo sample"
	@echo "make test       - run the pytest suite"
	@echo "make bench      - score every controller on the nominal plant"
	@echo "make optimum    - print the analytic and numerical time-optimal solution"
	@echo "make dashboard  - launch the Streamlit UI"

install:
	$(PY) -m pip install -e ".[dev,dashboard]"

figures:
	$(PY) run.py all

quick:
	$(PY) run.py all --quick

test:
	$(PY) -m pytest

bench:
	$(PY) run.py bench

optimum:
	$(PY) run.py optimum --numeric

dashboard:
	streamlit run src/dfp/dashboard/app.py

clean:                      # keeps results/results.json (it is committed)
	rm -rf .pytest_cache **/__pycache__ results/_studies
