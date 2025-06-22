#!/usr/bin/env python
from pathlib import Path
import matplotlib.pyplot as plt, numpy as np
from diafiltration.disturbance import run_mismatch, run_leak
from diafiltration.constants    import MP, cP_star, cL_star, kM_L

figdir = Path("figs"); figdir.mkdir(exist_ok=True)

# -------- param mismatch --------------------------
factors = (0.75, 0.50, 0.25)
for f in factors:
    t, peak, ok = run_mismatch(f)
    status = "✓" if ok else "✗"
    print(f"kM_L,true = {f:.2f}·kM_L  →  t={t:.2f} h, peak cL={peak:.1f}  {status}")

# -------- protein leakage --------------------------
t,cP_end,cL_end,ok = run_leak()
print(f"Leakage: finished={ok}, t={t:.2f} h, final cP={cP_end:.1f}, cL={cL_end:.1f}")
