#!/usr/bin/env python
from pathlib import Path
import matplotlib.pyplot as plt
from diafiltration import closed_loop, MP, cP_star, cL_star

t,V,ML,u = closed_loop()
cP = MP / V; cL = ML / V

figdir = Path("figs"); figdir.mkdir(exist_ok=True)
fig,ax = plt.subplots(3,1,figsize=(6,8),sharex=True)
ax[0].plot(t/3600,cP); ax[0].axhline(cP_star,ls="--"); ax[0].set_ylabel("c_P")
ax[1].plot(t/3600,cL); ax[1].axhline(cL_star,ls="--"); ax[1].set_ylabel("c_L")
ax[2].step(t[:-1]/3600,u,where="post"); ax[2].set_ylabel("u"); ax[2].set_xlabel("h")
fig.tight_layout(); fig.savefig(figdir/"nominal.png",dpi=150)
print("Saved figs/nominal.png")
