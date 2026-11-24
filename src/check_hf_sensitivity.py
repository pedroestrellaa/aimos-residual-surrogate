"""Step 4-fix - Sensitivity of the truth to the high-frequency radiation damping (2.2-4.0 rad/s).

The damping above 2.2 rad/s is single-body, fine mesh, with a radiation-vs-Haskind discrepancy of 20-30% and a 13-21%
jump at the join. It is bracketed by scaling it x0.5 and x2.
Criterion fixed BEFORE running: the 90th percentile over cases of |r_sigma(x0.5 or x2) - r_sigma(x1)| < 0.02,
r_sigma = ln(sigma_NL / sigma_Lr). Cases: 40 from the dataset design, half with soft connectors (log10 k < 5.9),
where the connector mode sits in 2.2-4 rad/s.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cummins_td import build_model, simulate  # noqa: E402
from make_dataset import design  # noqa: E402


def run(a):
    c, s = a
    r = simulate(build_model(c["gap"], hf_scale=s), c["Hs"], c["Tp"], 10 ** c["log10k"], c["seed"])
    return c["case"], s, np.log(r["F_td"].std() / r["F_fd"].std()), np.log(np.abs(r["F_td"]).max() / np.abs(r["F_fd"]).max())


if __name__ == "__main__":
    cases = design(1200)
    soft = [c for c in cases if c["log10k"] < 5.9][:20]
    stiff = [c for c in cases if c["log10k"] >= 5.9][:20]
    jobs = [(c, s) for c in soft + stiff for s in (0.5, 1.0, 2.0)]
    with Pool(30) as p:
        res = p.map(run, jobs)
    R = {(i, s): (rs, rm) for i, s, rs, rm in res}
    ids = [c["case"] for c in soft + stiff]
    for lab, grp in [("soft (log10k < 5.9)", soft), ("stiff", stiff), ("all", soft + stiff)]:
        g = [c["case"] for c in grp]
        ds = [abs(R[(i, s)][0] - R[(i, 1.0)][0]) for i in g for s in (0.5, 2.0)]
        dm = [abs(R[(i, s)][1] - R[(i, 1.0)][1]) for i in g for s in (0.5, 2.0)]
        print(f"{lab:<22} |dr_sigma| p50 {np.median(ds):.4f}  p90 {np.percentile(ds, 90):.4f}  max {max(ds):.4f}"
              f"   |dr_max| p50 {np.median(dm):.4f}  p90 {np.percentile(dm, 90):.4f}")
    p90 = np.percentile([abs(R[(i, s)][0] - R[(i, 1.0)][0]) for i in ids for s in (0.5, 2.0)], 90)
    print("S1 HF-damping sensitivity, p90 |dr_sigma| < 0.02:", f"{p90:.4f}", "PASS" if p90 < 0.02 else "FAIL")
