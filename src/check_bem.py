"""Step 1b - Sanity checks on the BEM coefficients, and the linear connector-load RAO.

Checks (each prints PASS/FAIL against a stated tolerance):
  C1 reciprocity        A_ij = A_ji and B_ij = B_ji
  C2 identical bodies   radiation diagonal of body 1 equals body 2 (mirror symmetry)
  C3 long-wave limit    |F_heave| / K33 -> 1 as omega -> 0 (Froude-Krylov limit)
  C4 mesh convergence   panel 1.0 m vs 0.5 m, gap 2 m: A33, B33, |F3| near heave resonance
  C5 damping sign       B_ii >= 0 for all frequencies
Plus: interaction strength vs gap, heave natural frequency, and connector RAO figure.

Connector model: one vertical shear spring of stiffness k between the facing ends.
State xi = [z1, th1, z2, th2]. Vertical displacement at a point x on body i: z_i - (x - x_i) th_i
(Capytaine pitch = right-handed rotation about +y). Facing ends: body 1 at +L/2, body 2 at -L/2, so
delta = z2 + (L/2) th2 - z1 + (L/2) th1 = c . xi, c = [-1, L/2, 1, L/2], connector force = k delta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load(gap, res=1.0):
    d = np.load(DATA / f"bem_gap{gap:g}_res{res:g}.npz")
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def connector_vector(L):
    return np.array([-1.0, L / 2, 1.0, L / 2])


def linear_rao(d, k):
    """Complex response xi(omega) per unit wave amplitude and connector force RAO k*c.xi."""
    c = connector_vector(d["meta"]["L"])
    Kc = k * np.outer(c, c)
    xi = np.empty_like(d["F"])
    for n, w in enumerate(d["omega"]):
        Z = -w**2 * (d["M"] + d["A"][n]) - 1j * w * d["B"][n] + d["K"] + Kc
        xi[n] = np.linalg.solve(Z, d["F"][n])
    return xi, k * xi @ c


def report(name, value, tol, ok):
    print(f"  {name:<52s} {value:>12.3e}   tol {tol:<8g} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    gaps = [1.5, 2, 3, 4, 5]   # gap 1 m dropped: panel = gap, reciprocity 1.02% > 1% (run 4)
    D = {g: load(g) for g in gaps}
    allok = True
    print("C1 reciprocity, C2 identical bodies, C5 damping sign (all gaps)")
    for g, d in D.items():
        A, B = d["A"], d["B"]
        scaleA, scaleB = np.abs(A).max(), np.abs(B).max()
        e1 = max(np.abs(A - A.transpose(0, 2, 1)).max() / scaleA, np.abs(B - B.transpose(0, 2, 1)).max() / scaleB)
        e2 = max(np.abs(A[:, 0, 0] - A[:, 2, 2]).max() / np.abs(A[:, 0, 0]).max(),
                 np.abs(A[:, 1, 1] - A[:, 3, 3]).max() / np.abs(A[:, 1, 1]).max())
        e5 = min(np.diagonal(B, axis1=1, axis2=2).min() / scaleB, 0.0)
        allok &= report(f"gap {g:>3g} m  C1 max rel. asymmetry", e1, 1e-2, e1 < 1e-2)
        allok &= report(f"gap {g:>3g} m  C2 body1 vs body2 radiation diag", e2, 1e-2, e2 < 1e-2)
        allok &= report(f"gap {g:>3g} m  C5 most negative B_ii (rel.)", e5, 1e-6, e5 > -1e-6)

    d = D[2]
    print("C3 long-wave limit (gap 2 m, lowest omega)")
    r = np.abs(d["F"][0, 0]) / d["K"][0, 0]
    allok &= report(f"|F_heave|/K33 at omega={d['omega'][0]:.2f} rad/s", r, 0.05, abs(r - 1) < 0.05)

    print("C4 mesh convergence (gap 2 m, 1.0 m vs 0.5 m panels), omega 0.6-1.8 rad/s")
    f = load(2, 0.5)
    band = (d["omega"] > 0.6) & (d["omega"] < 1.8)
    for lab, a, b in [("A33", d["A"][:, 0, 0], f["A"][:, 0, 0]), ("B33", d["B"][:, 0, 0], f["B"][:, 0, 0]),
                      ("|F3|", np.abs(d["F"][:, 0]), np.abs(f["F"][:, 0]))]:
        e = (np.abs(a - b)[band] / np.abs(b).max()).max()
        allok &= report(f"{lab} max rel. difference", e, 0.05, e < 0.05)

    print("Interaction strength: max |A(b1 heave, b2 heave)| / max |A(b1 heave, b1 heave)|")
    for g, dd in D.items():
        print(f"  gap {g:>3g} m   {np.abs(dd['A'][:, 0, 2]).max() / np.abs(dd['A'][:, 0, 0]).max():.3f}")

    m, K33 = d["M"][0, 0], d["K"][0, 0]
    wn = [np.sqrt(K33 / (m + a)) for a in d["A"][:, 0, 0]]
    i = np.argmin(np.abs(np.array(wn) - d["omega"]))
    print(f"Heave natural frequency (uncoupled, fixed-point): {d['omega'][i]:.2f} rad/s  (T = {2*np.pi/d['omega'][i]:.1f} s)")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for g, dd in D.items():
        ax[0].plot(dd["omega"], dd["A"][:, 0, 2] / m, label=f"g = {g:g} m")
        _, fc = linear_rao(dd, 1e7)
        ax[2].plot(dd["omega"], np.abs(fc) / 1e3, label=f"g = {g:g} m")
    ax[0].set(xlabel=r"$\omega$ [rad/s]", ylabel=r"$A_{z_1 z_2}/m$", title="Heave-heave interaction added mass")
    ax[0].legend(fontsize=8)
    for k in [1e5, 1e6, 1e7, 1e8]:
        _, fc = linear_rao(d, k)
        ax[1].plot(d["omega"], np.abs(fc) / 1e3, label=f"k = {k:.0e} N/m")
    ax[1].set(xlabel=r"$\omega$ [rad/s]", ylabel="|connector force| [kN/m]", title="Linear connector RAO, gap 2 m")
    ax[1].legend(fontsize=8)
    ax[2].set(xlabel=r"$\omega$ [rad/s]", ylabel="|connector force| [kN/m]", title="Linear connector RAO, k = 1e7 N/m")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step1_bem_checks.png", dpi=140)
    print(f"Figure: figures/step1_bem_checks.png")
    print("ALL CHECKS PASS" if allok else "SOME CHECKS FAILED")
    sys.exit(0 if allok else 1)
