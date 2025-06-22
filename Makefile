venv:
	python -m venv .venv && . .venv/Scripts/activate && pip install -e .[dev]
test:
	. .venv/Scripts/activate && pytest -q
figs:
	. .venv/Scripts/activate && python scripts/run_nominal.py
clean:
	del /s /q __pycache__ *.pyc
