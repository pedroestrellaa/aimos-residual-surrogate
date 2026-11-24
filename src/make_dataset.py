"""Step 4 - Dataset: potential-flow features + synthetic nonlinear truth for N cases.

Sampling (fixed before generation): Latin hypercube on Hs in [0.5, 4] m, Tp in [6, 12] s, log10 k in [5, 7];
gap cycles through {1.5, 2, 3, 4, 5} m. Case i uses wave seed 10_000 + i. Master seed 20261124.

Columns
  inputs        Hs, Tp, gap, log10k
  pf features   sigma_L, Tz_L, Emax_L      potential flow, spectral (available without any time simulation)
  truth         sigma_NL, max_NL           nonlinear Cummins model, 1800 s
  same-sea lin  sigma_Lr, max_Lr           linear FD superposition on the identical wave realisation
Residual targets (step 5): r_sigma = ln(sigma_NL / sigma_Lr), r_max = ln(max_NL / max_Lr).

Usage: python src/make_dataset.py --n 1200 --workers 28
"""
import argparse
import csv
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cummins_td import build_model, simulate, DELTA0, CD, CG_COEF  # noqa: E402
from linear_fd import linear_stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GAPS = [1.5, 2.0, 3.0, 4.0, 5.0]
COLS = ["case", "seed", "Hs", "Tp", "gap", "log10k", "sigma_L", "Tz_L", "Emax_L",
        "sigma_NL", "max_NL", "sigma_Lr", "max_Lr", "seconds"]
_MODELS = {}


def lhs(n, d, rng):
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.uniform(size=(n, d))) / n
    return u


def design(n):
    rng = np.random.default_rng(20261124)
    u = lhs(n, 3, rng)
    Hs = 0.5 + 3.5 * u[:, 0]; Tp = 6 + 6 * u[:, 1]; lk = 5 + 2 * u[:, 2]
    gap = np.array([GAPS[i % 5] for i in range(n)])
    return [dict(case=i, seed=10_000 + i, Hs=Hs[i], Tp=Tp[i], gap=gap[i], log10k=lk[i]) for i in range(n)]


def run(c):
    t0 = time.time()
    if c["gap"] not in _MODELS:
        _MODELS[c["gap"]] = build_model(c["gap"])
    k = 10 ** c["log10k"]
    lin = linear_stats(c["Hs"], c["Tp"], c["gap"], k)
    r = simulate(_MODELS[c["gap"]], c["Hs"], c["Tp"], k, c["seed"])
    return dict(c, sigma_L=lin["sigma_F"], Tz_L=lin["Tz"], Emax_L=lin["Emax_F"],
                sigma_NL=r["F_td"].std(), max_NL=np.abs(r["F_td"]).max(),
                sigma_Lr=r["F_fd"].std(), max_Lr=np.abs(r["F_fd"]).max(), seconds=time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--workers", type=int, default=28)
    a = ap.parse_args()
    cases = design(a.n)
    print(f"truth coefficients: CD={CD} DELTA0={DELTA0} CG_COEF={CG_COEF}", flush=True)
    out = ROOT / "data" / "dataset.csv"
    t0 = time.time(); rows = []
    with Pool(a.workers) as p:
        for i, r in enumerate(p.imap_unordered(run, cases, chunksize=4)):
            rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"{i+1}/{a.n} cases  {time.time()-t0:.0f} s", flush=True)
    rows.sort(key=lambda r: r["case"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.6g}" if isinstance(r[k], float) else r[k]) for k in COLS})
    print(f"wrote {out.name}: {len(rows)} rows in {time.time()-t0:.0f} s")
