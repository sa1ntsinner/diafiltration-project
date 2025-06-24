# Diafiltration — Time-Optimal MPC  
<sub><em>Advanced Process Control, SoSe 2025 • TU Dortmund</em></sub>

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue?logo=python)](https://www.python.org/) 
[![Licence BSD-3 (TU Dortmund)](https://img.shields.io/badge/license-BSD--3--Clause-green)](#-licence)

<div align="center">

**Non-linear batch diafiltration model** ⨯ **Time-optimal MPC (CasADi)**  
Robustness to **disturbance & plant-model mismatch** is built-in.

</div>

---

## ✨ Features

|   | Module | Description |
|---|--------|-------------|
| ✅ | `model.py` | Non-linear dynamics + RK4 integrator |
| ✅ | `mpc.py` | Time-optimal MPC with constraints |
| ✅ | `simulator.py` | Open/closed-loop simulation engine |
| ✅ | `tests.py` | Robustness: disturbance + param mismatch |
| ✅ | `views.py` | Streamlit frontend logic (MPC, tests, open-loop) |
| ✅ | `app.py` | Streamlit router & layout |
| ✅ | `constants.py` | Parameters used globally |

---

## 🚀 Quickstart

```bash
git clone https://github.com/sa1ntsinner/diafiltration-mpc.git
cd diafiltration-mpc

# create and activate the environment
conda env create -f environment.yml
conda activate DFP

# launch the Streamlit UI
streamlit run src/app.py
```

---

## 🖥️ Streamlit Interface

```text
src/
├─ app.py              ← Sidebar navigation & layout
├─ constants.py        ← Model constants (V0, MP, c*_L etc.)
├─ model.py            ← RK4 integrator and right-hand side
├─ mpc.py              ← Time-optimal MPC builder (CasADi)
├─ simulator.py        ← Open-loop & closed-loop logic
├─ tests.py            ← Disturbance & mismatch test functions
├─ views.py            ← Interactive Streamlit views per tab
└─ assets/
   └─ tank_image.png   ← Visual sketch for sidebar
```

---

## 🔧 Functionality

| Page       | Description |
|------------|-------------|
| 🟠 Open-loop | Try 1–5 constant `u` values and compare results |
| 🔵 MPC        | Run time-optimal MPC with horizon slider |
| 🧪 Test       | Visualise plant-model mismatch + disturbance robustness |

All simulations are interactive and respond live to user input.

---

## 📜 Citing
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
This student project is part of the course **Advanced Process Control**, TU Dortmund (2025).  
Licensed under BSD-3-Clause.