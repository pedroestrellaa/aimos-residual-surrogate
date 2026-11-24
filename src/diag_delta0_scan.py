"""Diagnostic: choose the hardening length DELTA0 so the synthetic truth stays a correction to potential flow.
Criterion fixed before the scan: sigma_NL/sigma_L in [0.6, 1.8] at every corner of the design space
(Hs 4 m worst case; Tp 6/12 s; gap 1.5/5 m; k 1e5/1e7 N/m). Smallest DELTA0 that meets it is chosen
(strongest nonlinearity that is still a correction)."""
import sys; from pathlib import Path; from itertools import product; from multiprocessing import Pool
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cummins_td as ct

def run(a):
    d0, g, Tp, k = a
    ct.DELTA0 = d0
    r = ct.simulate(ct.build_model(g), 4.0, Tp, k, seed=11)
    return a, r["F_td"].std() / r["F_fd"].std(), abs(r["F_td"]).max() / abs(r["F_fd"]).max()

if __name__ == "__main__":
    cases = list(product([0.5, 0.75, 1.0, 1.5], [1.5, 5], [6.0, 12.0], [1e5, 1e7]))
    with Pool(32) as p: res = p.map(run, cases)
    for d0 in [0.5, 0.75, 1.0, 1.5]:
        rs = [r for r in res if r[0][0] == d0]
        lo, hi = min(r[1] for r in rs), max(r[1] for r in rs)
        print(f"DELTA0 = {d0:<5}  sigma ratio range [{lo:.3f}, {hi:.3f}]   max ratio range [{min(r[2] for r in rs):.3f}, {max(r[2] for r in rs):.3f}]   {'MEETS' if 0.6 <= lo and hi <= 1.8 else 'fails'}")
        for (d, g, Tp, k), s, m in rs: print(f"     gap {g:<4} Tp {Tp:<5} k {k:.0e}: sigma {s:.3f}  max {m:.3f}")
