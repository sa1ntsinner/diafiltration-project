"""
dfp.cli
=======
Command-line front end – the reproducible path from source to report.

Examples
--------
::

    python -m dfp.cli list                 # show the available studies
    python -m dfp.cli all                  # regenerate every figure + results.json
    python -m dfp.cli all --quick          # same, with reduced Monte-Carlo sample
    python -m dfp.cli run tear objectives  # one or several studies
    python -m dfp.cli optimum              # print the analytic / numerical optimum
    python -m dfp.cli bench --horizon 20   # closed-loop timing of one controller
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List

import numpy as np


def _cmd_list(_args) -> int:
    from .experiments import STUDIES

    print("available studies:\n")
    for name, fn in STUDIES.items():
        doc = (fn.__doc__ or "").strip().splitlines()[0]
        print(f"  {name:<12s} {doc}")
    return 0


def _cmd_all(args) -> int:
    from .experiments import run_all

    run_all(quick=args.quick, isolate=not getattr(args, "in_process", False),
            skip_existing=getattr(args, "skip_existing", False))
    return 0


def _cmd_run(args) -> int:
    from .experiments import run_all

    run_all(args.studies, quick=args.quick,
            isolate=not getattr(args, "in_process", False),
            skip_existing=getattr(args, "skip_existing", False))
    return 0


def _cmd_optimum(args) -> int:
    from .config import NOMINAL
    from .controllers import analytic_optimum, solve_min_time_ocp

    a = analytic_optimum(NOMINAL)
    print("analytic time-optimal solution")
    print(f"  structure                : {'bang-bang (u=0 then u=1)' if a.bang_bang else 'singular arc present'}")
    print(f"  pre-concentration ends at: {a.t_switch/3600:.5f} h  (cL = {a.cL_switch:.2f} mol/m^3)")
    print(f"  optimal batch time       : {a.T_h:.5f} h")
    print(f"  peak lactose             : {a.cL_peak:.2f} mol/m^3  (limit {NOMINAL.cL_max:g})")
    if args.numeric:
        ocp = solve_min_time_ocp(NOMINAL, N=args.intervals)
        print(f"\ndirect OCP with N = {args.intervals}")
        print(f"  optimal batch time       : {ocp['T_h']:.5f} h")
        print(f"  relative deviation       : {abs(ocp['T_h']-a.T_h)/a.T_h:.2e}")
    return 0


def _cmd_bench(args) -> int:
    from .config import NOMINAL
    from .controllers import (BangBang, ThresholdPolicy, analytic_optimum,
                              build_multistage_nmpc, build_nmpc)
    from .plant import nominal_plant
    from .simulate import closed_loop

    a = analytic_optimum(NOMINAL)
    ctrls = {
        "analytic bang-bang": BangBang(NOMINAL),
        "threshold policy": ThresholdPolicy(params=NOMINAL),
        f"tracking MPC N={args.horizon}": build_nmpc("tracking", args.horizon),
        f"l1-time MPC N={args.horizon}": build_nmpc("l1_time", args.horizon),
        f"min-time MPC N={args.horizon}": build_nmpc("min_time", args.horizon),
        f"multi-stage NMPC N={args.horizon}": build_multistage_nmpc(args.horizon),
    }
    print(f"{'controller':<32s}{'T [h]':>9s}{'gap [%]':>9s}"
          f"{'cP end':>9s}{'peak cL':>9s}{'solve [ms]':>12s}")
    print("-" * 80)
    print(f"{'time-optimal (analytic)':<32s}{a.T_h:>9.3f}{0.0:>9.2f}"
          f"{NOMINAL.cP_f:>9.1f}{a.cL_peak:>9.0f}{'-':>12s}")
    for name, c in ctrls.items():
        r = closed_loop(c, nominal_plant(), label=name)
        gap = 100 * (r.batch_time_h - a.T_h) / a.T_h if r.finished else float("nan")
        ms = float(np.mean(r.solve_times)) * 1e3 if r.solve_times.size else 0.0
        print(f"{name:<32s}{r.batch_time_h:>9.3f}{gap:>9.2f}"
              f"{float(r.cP[-1]):>9.1f}{r.cL_peak:>9.0f}{ms:>12.1f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dfp.cli",
        description="Time-optimal control of a batch diafiltration process "
                    "(APC SS25, TU Dortmund).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list the available studies").set_defaults(func=_cmd_list)

    a = sub.add_parser("all", help="run every study and write docs/figures + results")
    a.add_argument("--quick", action="store_true", help="smaller Monte-Carlo sample")
    a.add_argument("--in-process", action="store_true",
                   help="do not isolate the studies in subprocesses")
    a.add_argument("--skip-existing", action="store_true",
                   help="reuse studies already cached in results/_studies")
    a.set_defaults(func=_cmd_all)

    r = sub.add_parser("run", help="run selected studies")
    r.add_argument("studies", nargs="+")
    r.add_argument("--quick", action="store_true")
    r.add_argument("--in-process", action="store_true")
    r.add_argument("--skip-existing", action="store_true")
    r.set_defaults(func=_cmd_run)

    o = sub.add_parser("optimum", help="print the time-optimal solution")
    o.add_argument("-n", "--numeric", action="store_true",
                   help="also solve the direct OCP")
    o.add_argument("--intervals", type=int, default=200)
    o.set_defaults(func=_cmd_optimum)

    m = sub.add_parser("merge", help="rebuild results.json from the study cache")
    m.set_defaults(func=lambda _a: (__import__(
        "dfp.experiments.studies", fromlist=["_merge_cache"])._merge_cache(False), 0)[1])

    b = sub.add_parser("bench", help="benchmark all controllers on the nominal plant")
    b.add_argument("--horizon", type=int, default=20)
    b.set_defaults(func=_cmd_bench)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
