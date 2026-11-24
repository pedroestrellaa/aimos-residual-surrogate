"""Step 3b - Gate checks for the time-domain model, and the effect of each nonlinear term.

G1 kernel quality (all gaps):   Ogilvie A_inf spread < 3%, A_inf vs BEM A_inf < 3%, kernel tail (last 5 s) < 2%
G2 linear gate (12 cases):      all nonlinear terms OFF, 1800 s, same wave realisation:
                                |sigma_TD/sigma_FD - 1| < 3%   and   RMS(F_TD - F_FD)/sigma_FD < 5%
G3 time-step convergence:       all terms ON, dt 0.05 vs 0.025 s: |sigma ratio - 1| < 2%, |max ratio - 1| < 3%
G4 term-by-term effect (reported, no tolerance): sigma_NL / sigma_L and max_NL / max_L for each term alone and all on
"""
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cummins_td import build_model, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAPS = [1.5, 2, 3, 4, 5]


def report(name, value, tol, ok):
    print(f"  {name:<64s} {value:>10.4f}   tol {tol:<6g} {'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_case(args):
    gap, Hs, Tp, k, seed, dt, flags = args
    m = build_model(gap, dt=dt)
    r = simulate(m, Hs, Tp, k, seed, **flags)
    return dict(args=args, s_td=r["F_td"].std(), s_fd=r["F_fd"].std(),
                m_td=np.abs(r["F_td"]).max(), m_fd=np.abs(r["F_fd"]).max(),
                rms=(r["F_td"] - r["F_fd"]).std() / r["F_fd"].std(),
                F_td=r["F_td"][:3000], F_fd=r["F_fd"][:3000], t=r["t"][:3000])


OFF = dict(drag=False, hardening=False, gap_damping=False)
ON = dict(drag=True, hardening=True, gap_damping=True)

if __name__ == "__main__":
    t0 = time.time(); ok = True
    print("G1 kernel quality")
    for g in GAPS:
        mt = build_model(g).meta
        ok &= report(f"gap {g} m  Ogilvie A_inf spread", mt["A_inf_ogilvie_spread"], 0.03, mt["A_inf_ogilvie_spread"] < 0.03)
        ok &= report(f"gap {g} m  A_inf vs BEM", mt["A_inf_vs_bem"], 0.03, mt["A_inf_vs_bem"] < 0.03)
        ok &= report(f"gap {g} m  kernel tail", mt["K_tail"], 0.02, mt["K_tail"] < 0.02)

    with Pool(28) as pool:
        print("G2 linear gate: nonlinear terms OFF, 1800 s, identical wave realisation")
        cases = [(g, 2.0, Tp, k, 100 + i, 0.05, OFF)
                 for i, (g, k, Tp) in enumerate(product([1.5, 5], [1e5, 1e6, 1e7], [6.0, 12.0]))]
        for r in pool.map(run_case, cases):
            g, _, Tp, k = r["args"][:4]
            e1 = abs(r["s_td"] / r["s_fd"] - 1)
            ok &= report(f"gap {g} k {k:.0e} Tp {Tp:>4}: |sigma ratio - 1|", e1, 0.03, e1 < 0.03)
            ok &= report(f"gap {g} k {k:.0e} Tp {Tp:>4}: RMS diff / sigma", r["rms"], 0.05, r["rms"] < 0.05)
        lin_example = r

        print("G3 time-step convergence: all terms ON, dt 0.05 vs 0.025 s")
        cc = [(1.5, 3.0, 6.0, 1e7, 7, 0.05, ON), (1.5, 3.0, 6.0, 1e7, 7, 0.025, ON),
              (5, 3.0, 8.0, 1e5, 8, 0.05, ON), (5, 3.0, 8.0, 1e5, 8, 0.025, ON)]
        rr = pool.map(run_case, cc)
        for a, b in [(rr[0], rr[1]), (rr[2], rr[3])]:
            g, _, Tp, k = a["args"][:4]
            e = abs(a["s_td"] / b["s_td"] - 1); em = abs(a["m_td"] / b["m_td"] - 1)
            ok &= report(f"gap {g} k {k:.0e}: |sigma(0.05)/sigma(0.025) - 1|", e, 0.02, e < 0.02)
            ok &= report(f"gap {g} k {k:.0e}: |max(0.05)/max(0.025) - 1|", em, 0.03, em < 0.03)

        print("G4 effect of each nonlinear term (Hs 3 m, Tp 7 s): sigma_NL/sigma_L , max_NL/max_L")
        terms = {"drag only": dict(drag=True, hardening=False, gap_damping=False),
                 "hardening only": dict(drag=False, hardening=True, gap_damping=False),
                 "gap damping only": dict(drag=False, hardening=False, gap_damping=True),
                 "all on": ON}
        combos = [(name, g, k) for name in terms for g in [1.5, 5] for k in [1e5, 1e7]]
        res = pool.map(run_case, [(g, 3.0, 7.0, k, 300, 0.05, terms[name]) for name, g, k in combos])
        for (name, g, k), r in zip(combos, res):
            print(f"  {name:<17s} gap {g:<4} k {k:.0e}:  sigma {r['s_td']/r['s_fd']:.3f}   max {r['m_td']/r['m_fd']:.3f}")
        nl_example = res[-4]   # all on, gap 1.5, k 1e5

    fig, ax = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True)
    for a_, r, ttl in [(ax[0], lin_example, "Gate G2: nonlinear terms OFF"), (ax[1], nl_example, "Synthetic truth: all terms ON")]:
        a_.plot(r["t"], r["F_fd"] / 1e3, lw=1.2, label="linear frequency domain (potential flow)")
        a_.plot(r["t"], r["F_td"] / 1e3, lw=0.9, ls="--", label="Cummins time domain")
        g, Hs, Tp, k = r["args"][:4]
        a_.set(ylabel="connector force [kN]", title=f"{ttl}  (gap {g} m, Hs {Hs} m, Tp {Tp} s, k {k:.0e} N/m)")
        a_.legend(fontsize=8, loc="upper right")
    ax[1].set_xlabel("t [s]"); ax[1].set_xlim(r["t"][0], r["t"][0] + 150)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "step3_time_domain.png", dpi=140)
    print(f"Figure: figures/step3_time_domain.png   wall time {time.time()-t0:.0f} s")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
