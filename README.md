# 🧪 Diafiltration Process MPC
<sub><em>Advanced Process Control • SoSe 2025 • TU Dortmund</em></sub>

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue?logo=python)](https://www.python.org/)

<div align="left">

- 🔄 Non-linear **batch diafiltration model** with *volumetric* + *concentration dynamics*
- 🧠 Time-optimal **MPC implementation** using **CasADi** for *symbolic optimization*
- 🛡️ Robust to **disturbances** and **plant-model mismatches**
- 🌐 Streamlit-based *interface* for **interactive control experiments** and **benchmarking**

</div>

---

## ✨ Overview

This project implements and evaluates **non-linear Model Predictive Control (MPC)** strategies for a **diafiltration process**. A Streamlit-based UI enables full simulation and comparison of:

- **Constant open-loop control**
- **Heuristic threshold policy**
- **Time-optimal MPC**
- **Economic MPC with time-of-use energy tariff**
- **Robust MPC under uncertainty**

All simulations are dynamic and interactive.

---

## 📂 Project Structure
```text
diafiltration-project/
├── assets/
│   └── tank_image.png              # Sidebar image for Streamlit
│
├── src/
│   ├── app.py                      # Streamlit entrypoint with sidebar & routing
│   ├── views.py                    # UI logic for Open-loop, MPC, Tests tabs
│
│   ├── control/                    # MPC builder modules
│   │   ├── __init__.py             # Imports robust/builder factories
│   │   ├── builder.py              # build_mpc(): MPC setup (CasADi)
│   │   ├── robust.py               # Robust MPC formulation
│
│   ├── core/                       # Core model and numerical routines
│   │   ├── discretise.py           # RK4 integrator for simulation
│   │   ├── dynamics.py             # Diafiltration RHS model
│   │   ├── linearise.py            # Continuous-time Jacobians (A, B)
│   │   ├── params.py               # Global parameters (MP, V0, cP_star etc.)
│   │   ├── tariff.py               # Time-of-use electricity cost function
│
│   ├── experiments/                # Extra simulation tools
│   │   ├── montecarlo.py           # Monte-Carlo robustness test logic
│
│   ├── sim/                        # Simulation wrappers
│   │   ├── __init__.py             # Imports simulate, mpc_* controllers
│   │   ├── simulate.py             # Core simulation loop (simulate())
│   │   ├── scenarios.py            # Nominal, tear, mismatch, leakage etc.
│
├── .gitignore                      # Ignore cache, .env, IDEs, results
├── environment.yml                 # Conda environment with pinned packages
├── requirements.txt                # pip-compatible dependency list
├── README.md                       # Project documentation
```

---

## ✅ Features

| ✅ | Module         | Description |
|----|----------------|-------------|
| ✔️ | `model.py`     | RK4 simulation of non-linear batch diafiltration |
| ✔️ | `mpc.py`       | Time-optimal MPC with terminal constraints |
| ✔️ | `simulator.py` | Unified open- and closed-loop simulator |
| ✔️ | `views.py`     | Streamlit dashboard with 3 interactive tabs |
| ✔️ | `tests.py`     | Evaluation under disturbances and uncertainties |
| ✔️ | `tariff.py`    | Piecewise time-of-use electricity cost model |
| ✔️ | `montecarlo.py`| Batch-wise robustness testing via random plant draws |

---

## 🧑‍💻 Quickstart

```bash
git clone https://github.com/sa1ntsinner/diafiltration-project.git
cd diafiltration-project

# Set up environment
conda env create -f environment.yml --name DFP
conda activate DFP

# Launch UI
streamlit run src/app.py
```

---

## 🖥️ Streamlit Dashboard
🟠 Open-loop Simulation
Test and compare constant u values (1–5 options). Visualise their effect on:
Product concentration ($c_P$)
Contaminant level ($c_L$)
Batch volume ($V$)

🔵 MPC Showcase
Compare controller types:
Spec-tracking MPC
Threshold policy
Time-optimal MPC
Economic MPC (TOU electricity tariff)

🧪 Robustness Testing
Evaluate MPC resilience under:
Filter-cake tears
Parameter mismatches (Km)
Protein leakage
Monte-Carlo stress testing with randomised plants

---

## 📜 Citing
```bibtex
@misc{mirzayev2025diafiltration,
  author       = {Mirzayev, Elmir and Dharan, Rakesh and Krishan, Kirupa},
  title        = {Diafiltration Process MPC},
  year         = {2025},
  howpublished = {\url{https://github.com/sa1ntsinner/diafiltration-project}},
  note         = {Advanced Process Control, TU Dortmund}
}
```
---

**Contributors:**  
🧑‍💻 Elmir Mirzayev ([sa1ntsinner](https://github.com/sa1ntsinner))  
🧑‍💻 Rakesh Dharan ([Rakeshdharan](https://github.com/Rakeshdharan))  
🧑‍💻 Kirupa Krishan ([kirupakrishan](https://github.com/kirupakrishan))

---

## © Licence
This student project is part of the course **Advanced Process Control**, TU Dortmund (2025).  
