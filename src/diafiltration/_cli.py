import argparse, matplotlib.pyplot as plt
from .simulator  import closed_loop
from .constants  import MP, cP_star, cL_star

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-N", type=int, default=20, help="MPC horizon")
    ns = ap.parse_args()

    t,V,ML,u = closed_loop(N=ns.N)
    cP = MP / V; cL = ML / V

    plt.figure(figsize=(6,5))
    ax1=plt.subplot(311); ax1.plot(t/3600,cP); ax1.axhline(cP_star,ls="--"); ax1.set_ylabel("c_P")
    ax2=plt.subplot(312); ax2.plot(t/3600,cL); ax2.axhline(cL_star,ls="--"); ax2.set_ylabel("c_L")
    ax3=plt.subplot(313); ax3.step(t[:-1]/3600,u,where="post"); ax3.set_ylabel("u"); ax3.set_xlabel("h")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
