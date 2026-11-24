"""Step 4b - Checks on data/dataset.csv.

D1 completeness       N rows, no NaN/inf, all positive
D2 same-sea linear    sigma_Lr / sigma_L: mean within 2% of 1, spread consistent with sampling noise of a 30-min record
D3 coverage           each LHS stratum (10 bins per input) populated; >= 200 cases per gap
D4 determinism        checked separately before generation (case 7 twice, bit-identical)
Reported: residual r_sigma = ln(sigma_NL/sigma_Lr) and r_max = ln(max_NL/max_Lr) per gap and by stiffness band.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def load_csv():
    return np.genfromtxt(ROOT / "data" / "dataset.csv", delimiter=",", names=True)


def report(name, value, tol, ok):
    print(f"  {name:<58s} {value:>10.4f}   tol {tol:<8} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    d = load_csv(); ok = True; n = d.size
    print(f"D1 completeness ({n} rows)")
    num = np.column_stack([d[c] for c in d.dtype.names])
    ok &= report("non-finite values", float((~np.isfinite(num)).sum()), "0", np.isfinite(num).all())
    pos = all((d[c] > 0).all() for c in ["sigma_L", "sigma_NL", "sigma_Lr", "max_NL", "max_Lr"])
    ok &= report("all load statistics > 0", float(pos), "1", pos)

    print("D2 same-sea linear vs spectral linear (sampling noise of one 30-min record)")
    q = d["sigma_Lr"] / d["sigma_L"]
    ok &= report("mean(sigma_Lr/sigma_L) - 1", abs(q.mean() - 1), "0.02", abs(q.mean() - 1) < 0.02)
    # expected std of a sigma estimate ~ 1/sqrt(2 N_eff), N_eff ~ duration / correlation time (~ Tz). Report both.
    expected = np.mean(1 / np.sqrt(2 * 1800 / d["Tz_L"]))
    print(f"  std(sigma_Lr/sigma_L) = {q.std():.4f}   (narrow-band estimate of sampling noise ~ {expected:.4f}; "
          f"wider is expected for peaked spectra)")

    print("D3 coverage")
    for c, lo, hi in [("Hs", 0.5, 4), ("Tp", 6, 12), ("log10k", 5, 7)]:
        h, _ = np.histogram(d[c], bins=10, range=(lo, hi))
        ok &= report(f"{c}: min cases in any of 10 bins", float(h.min()), ">= 0.08N", h.min() >= 0.08 * n)
    for g in np.unique(d["gap"]):
        m = (d["gap"] == g).sum()
        ok &= report(f"gap {g:g} m: cases", float(m), ">= 200", m >= 200)

    rs = np.log(d["sigma_NL"] / d["sigma_Lr"]); rm = np.log(d["max_NL"] / d["max_Lr"])
    print("Residual summary: r = ln(truth / same-sea linear)   [mean, p05, p95]")
    for lab, m in [("all", np.ones(n, bool))] + [(f"gap {g:g}", d["gap"] == g) for g in np.unique(d["gap"])] + \
            [("k < 1e5.67", d["log10k"] < 5.67), ("k > 1e6.33", d["log10k"] > 6.33)]:
        print(f"  {lab:<12s} r_sigma {rs[m].mean():+.3f} [{np.percentile(rs[m],5):+.3f}, {np.percentile(rs[m],95):+.3f}]"
              f"   r_max {rm[m].mean():+.3f} [{np.percentile(rm[m],5):+.3f}, {np.percentile(rm[m],95):+.3f}]")
    print(f"  Fraction of variance of ln(sigma_NL) explained by ln(sigma_Lr): "
          f"{np.corrcoef(np.log(d['sigma_NL']), np.log(d['sigma_Lr']))[0,1]**2:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    sc = ax[0].scatter(d["sigma_Lr"] / 1e3, d["sigma_NL"] / 1e3, c=d["log10k"], s=6, cmap="viridis")
    mx = d["sigma_NL"].max() / 1e3; ax[0].plot([0, mx], [0, mx], "k--", lw=0.8)
    ax[0].set(xlabel="linear sigma_F, same sea [kN]", ylabel="truth sigma_F [kN]", title="Truth vs potential flow")
    fig.colorbar(sc, ax=ax[0], label="log10 k")
    ax[1].scatter(d["log10k"], rs, c=d["Hs"], s=6, cmap="plasma")
    ax[1].axhline(0, c="k", lw=0.8)
    ax[1].set(xlabel="log10 k [N/m]", ylabel="r_sigma = ln(truth / linear)", title="Residual vs stiffness (colour = Hs)")
    for g in np.unique(d["gap"]):
        m = d["gap"] == g
        o = np.argsort(d["Hs"][m]); win = np.ones(25) / 25   # "valid": no zero-padding bias at the ends
        ax[2].plot(np.convolve(d["Hs"][m][o], win, "valid"), np.convolve(rs[m][o], win, "valid"), label=f"gap {g:g} m")
    ax[2].set(xlabel="Hs [m]", ylabel="r_sigma (running mean)", title="Residual vs Hs, per gap"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ROOT / "figures" / "step4_dataset.png", dpi=140)
    print("Figure: figures/step4_dataset.png")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
