"""
$ python -m diafiltration [-N 30]
Creates figs/nominal.png using closed_loop().
"""
import argparse, pathlib, matplotlib.pyplot as plt
from .simulator  import closed_loop
from .constants  import MP

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=20, help="prediction horizon")
    args = ap.parse_args()

    t, V, ML, u = closed_loop(N=args.N)
    cP, cL = MP/V, ML/V

    out = pathlib.Path("figs"); out.mkdir(exist_ok=True)
    fn  = out / "nominal.png"

    fig, ax = plt.subplots(3,1,figsize=(6,10),sharex=True)
    ax[0].plot(t/3600, cP); ax[0].set_ylabel("c_P [mol/m³]")
    ax[1].plot(t/3600, cL); ax[1].set_ylabel("c_L [mol/m³]")
    ax[2].step(t[:-1]/3600, u, where="post"); ax[2].set_ylabel("u")
    ax[2].set_xlabel("time [h]")
    fig.tight_layout(); fig.savefig(fn, dpi=150)
    print("saved", fn)

if __name__ == "__main__":
    main()
