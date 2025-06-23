from pathlib import Path
import matplotlib.pyplot as plt
from diafiltration.simulator import closed_loop
from diafiltration.constants  import MP

t, V, ML, u = closed_loop()
cP, cL = MP/V, ML/V
Path("figs").mkdir(exist_ok=True)
fn = Path("figs/nominal.png")

fig, ax = plt.subplots(3,1,figsize=(6,10),sharex=True)
ax[0].plot(t/3600, cP); ax[0].set_ylabel("c_P [mol/m³]")
ax[1].plot(t/3600, cL); ax[1].set_ylabel("c_L [mol/m³]")
ax[2].step(t[:-1]/3600, u, where="post"); ax[2].set_ylabel("u")
ax[2].set_xlabel("time [h]")
fig.tight_layout(); fig.savefig(fn, dpi=150)
print("saved", fn)
