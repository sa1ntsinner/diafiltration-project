# Diafiltration – Time-Optimal MPC  
*(Advanced Process Control, 2025 coursework)*  

[![CI](https://github.com/<your-username>/diafiltration-project/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/diafiltration-project/actions) 
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue?logo=python)](https://www.python.org/)

This repository delivers a **non-linear batch diafiltration model** and a  
**time-optimal Model-Predictive Controller (MPC)** implemented with CasADi.  
Extra modules analyse robustness to  

* parametric mismatch of the mass-transfer coefficient *k<sub>M,L</sub>* and  
* structural mismatch caused by protein leakage through the membrane.

---

## ✨ Key features

| ✔ | Module | Purpose |
|---|--------|---------|
| `model.py` | Non-linear membrane model + RK4 integrator (NumPy + CasADi) |
| `mpc.py` | Builds time-optimal MPC with path/terminal constraints |
| `simulator.py` | Closed-loop and open-loop plant simulation helpers |
| `disturbance.py` | Robustness study: _k<sub>M,L</sub>_ mismatch + protein leakage |
| `tests/` | PyTest suite (< 1 s) used in CI |
| GitHub Actions | Runs tests on every push (badge ↑) |

---

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/diafiltration-project.git
cd diafiltration-project

# create and activate virtual env
python -m venv .venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# install package in editable mode + dev deps
pip install -e .[dev]

# 1. unit & regression tests
pytest -q      # should print: ..

# 2. nominal closed-loop run (creates figs/nominal.png)
python scripts/run_nominal.py

---

## 📂 Project layout

src/diafiltration/        core library (import diafiltration as df)
│   constants.py          physical data & specs
│   model.py              flux functions, ODE RHS, RK4 helpers
│   mpc.py                build_mpc()
│   simulator.py          closed_loop(), open_loop()
│   disturbance.py        mismatch & leakage scenarios
│   _cli.py               python -m diafiltration [-N 30]
scripts/                  thin wrappers → generate figures
tests/                    PyTest regression + unit checks
notebooks/                exploration / final report plots
figs/                     auto-generated images (git-ignored)
reports/                  report.pdf, slides.pdf
.github/workflows/ci.yml  GitHub Actions (pytest on push)
Makefile                  make venv / make test / make figs
pyproject.toml            PEP 621 metadata (editable install)
