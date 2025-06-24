# Diafiltration — Time-Optimal MPC  
<sub><em>Advanced Process Control, SoSe 2025 • TU Dortmund</em></sub>

[![CI status](https://github.com/sa1ntsinner/diafiltration-project/actions/workflows/ci.yml/badge.svg)](https://github.com/sa1ntsinner/diafiltration-project/actions) 
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue?logo=python)](https://www.python.org/) 
[![Licence BSD-3 (TU Dortmund)](https://img.shields.io/badge/license-BSD--3--Clause-green)](#-licence)


<div align="center">

**Non-linear batch diafiltration model**   ⨯   **Time-optimal MPC (CasADi)**  
Robustness to **parameter & structural mismatch** built-in.

</div>

---

## ✨ Highlights

|   | Module | What it does |
|---|--------|--------------|
| ✅ | `model.py` | Non-linear diafiltration ODE + RK4 integrator (*NumPy & CasADi*) |
| ✅ | `mpc_controller.py` | Builds time-optimal MPC with path & terminal constraints |
| ✅ | `simulation.py` | Drop-in closed-loop / open-loop simulator |
| ✅ | `robustness.py` | Param-mismatch (*k*<sub>M,L</sub>) & protein-leakage scenarios |
| ✅ | `tests/` | < 1 s PyTest suite executed in CI |

---

## 🚀 Quick start

```bash
git clone https://github.com/sa1ntsinner/diafiltration-mpc.git
cd diafiltration-mpc

# create & activate conda env
conda env create -f environment.yml
conda activate DFP

# install in editable mode
python -m pip install -e .

# run nominal closed-loop experiment
python experiments/main.py
```

---

## 📂 Project layout
```text

diafiltration-mpc/
├─ .github/workflows/ci.yml        ← PyTest on push & PR
├─ .gitignore
├─ environment.yml                 ← Reproducible conda env
├─ README.md                       ← You are here
│
├─ src/                            ← Installable package "diafiltration_mpc"
│  ├─ __init__.py
│  ├─ parameters.py
│  ├─ model.py
│  ├─ policies.py
│  ├─ mpc_controller.py
│  ├─ simulation.py
│  ├─ plotting.py
│  └─ robustness.py
│
├─ experiments/                    ← Recreate all paper figures
│  ├─ main.py                      ← Nominal loop & baseline comparison
│  ├─ horizon_study.py             ← Influence of prediction horizon N
│  ├─ param_mismatch.py            ← k_M,L mismatch robustness
│  └─ structural_mismatch.py       ← Protein-leakage robustness
│
└─ tests/                          ← CI sanity checks
   ├─ test_model.py
   └─ test_mpc.py

```

---

## 🔧 Handy commands

| Action                                    | Command                                     |
| ----------------------------------------- | ------------------------------------------- |
| Regenerate main figure set                | `python experiments/main.py`                |
| Horizon sensitivity (N=5 … 50)            | `python experiments/horizon_study.py`       |
| Severe *k*<sub>M,L</sub> mismatch (0.25×) | `python experiments/param_mismatch.py`      |
| Structural leakage study                  | `python experiments/structural_mismatch.py` |
| Run all tests locally                     | `pytest -q`                                 |
| Build the LaTeX report                    | `make -C report`                            |


---

### 🧪 Continuous integration
```text
The GitHub Actions workflow **`ci.yml`** recreates _exactly the same_
conda environment we ship for local work (`environment.yml`, Python 3.9),
installs the project in editable mode and runs the full PyTest suite.  
A green badge means the commit is 100 % reproducible on a clean runner.
```

---

## 📜 Citing / authors
```bibtex
@misc{DiafiltrationMPC2025,
  author       = {Elmir Mirzayev},
  title        = {Diafiltration — Time‐Optimal MPC},
  howpublished = {\url{https://github.com/sa1ntsinner/diafiltration-mpc}},
  year         = {2025}
}
```

---

## © Licence
Project for Advanced Process Control 2025, TU Dortmund
