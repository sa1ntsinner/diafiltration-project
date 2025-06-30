# 🧪 Diafiltration Process MPC
<sub><em>Advanced Process Control • SoSe 2025 • TU Dortmund</em></sub>

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue?logo=python)](https://www.python.org/) 
[![License BSD-3 (TU Dortmund)](https://img.shields.io/badge/license-BSD--3--Clause-green)](#-license)

<div align="center">

🚰 Non-linear **batch diafiltration model**  
🧠 Advanced **Time-Optimal MPC** using **CasADi**  
🛡️ Robust to **disturbances** and **plant-model mismatches**

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
src/
├── app.py # Streamlit app launcher and navigation
├── constants.py # Global model parameters
├── model.py # Diafiltration dynamics (RHS) + RK4 integrator
├── mpc.py # MPC builder (CasADi formulation)
├── simulator.py # Open/closed-loop simulation logic
├── tests.py # Test scenarios (disturbance, mismatch, MC)
├── views.py # Streamlit UI logic (tabs)
├── assets/
│ └── tank_image.png # Sidebar illustration
├── core/, control/, sim/, experiments/
│ └── Modular control logic and scenario definitions
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
conda env create -f environment.yml
conda activate DFP

# Launch UI
streamlit run src/app.py
```

---

# 🖥️ Streamlit Dashboard
## 🟠 Open-loop Simulation
Test and compare constant u values (1–5 options). Visualise their effect on:
Product concentration ($c_P$)
Contaminant level ($c_L$)
Batch volume ($V$)

## 🔵 MPC Showcase
Compare controller types:
Spec-tracking MPC
Threshold policy
Time-optimal MPC
Economic MPC (TOU electricity tariff)

## 🧪 Robustness Testing
Evaluate MPC resilience under:
Filter-cake tears
Parameter mismatches (Km)
Protein leakage
Monte-Carlo stress testing with randomised plants

All simulations are interactive and respond live to user input.

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
Licensed under BSD-3-Clause.