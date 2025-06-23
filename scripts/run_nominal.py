"""
Generate figure of nominal closed-loop MPC batch (figs/nominal.png).
"""

import matplotlib.pyplot as plt
from pathlib import Path
from diafiltration import closed_loop, MP, cP_star, cL_star, cL_max

t,V,ML,u = closed_loop()
cP = MP / V
cL = ML / V

figdir = Path("figs"); figdir.mkdir(exist_ok=True)

fig,ax = plt.subplots(3,1,figsize=(6,10),sharex=True)
ax[0].plot(t/3600, cP); ax[0].axhline(cP_star,ls='--',color='k'); ax[0].set_ylabel('$c_P$')
ax[1].plot(t/3600, cL); ax[1].axhline(cL_star,ls='--',color='k'); ax[1].axhline(cL_max,ls=':',color='r')
ax[1].set_ylabel('$c_L$')
ax[2].step(t[:len(u)]/3600, u, where='post'); ax[2].set_ylabel('$u$'); ax[2].set_xlabel('Time [h]')
plt.tight_layout()
plt.savefig(figdir/'nominal.png', dpi=120)
print("Figure saved → figs/nominal.png")
