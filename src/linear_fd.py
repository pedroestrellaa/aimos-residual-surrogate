"""Step 2 - Linear frequency-domain connector load in irregular seas (the potential-flow prediction).

For a case (Hs, Tp, gap, k):
  H_F(omega)  connector force per unit wave amplitude, from the step-1 BEM coefficients (check_bem.linear_rao)
  S(omega)    JONSWAP spectrum, gamma = 3.3, normalised so that m0 = Hs^2/16 on a wide grid (0.05-6 rad/s)
  sigma_F   = sqrt( int |H_F|^2 S domega )                       over the BEM range 0.2-2.2 rad/s
  Tz        = 2 pi sqrt(m0_F / m2_F)                             mean zero-upcrossing period of the force
  E[max]    = sigma_F ( sqrt(2 ln N) + 0.5772 / sqrt(2 ln N) ),  N = duration / Tz   (Rayleigh peaks, narrow band)

Checks (tolerances fixed before running):
  L1 spectrum normalisation      4 sqrt(m0) on the wide grid = Hs                        (0.5%)
  L2 truncation                  wave energy outside 0.2-2.2 rad/s for Tp in [6, 12] s   (report; flag > 5%)
  L3 grid convergence            sigma_F on the 60-point BEM grid vs 600-point interp.   (2%)
  L4 Monte-Carlo (independent)   std of random-phase time series vs sigma_F               (3%, 20 x 3 h)
  L5 zero-stiffness limit        k -> 0 gives connector force -> 0                       (sigma_F/sigma_F(1e7) < 1e-3)
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_bem import load, linear_rao  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
GAMMA = 3.3
W_WIDE = np.linspace(0.05, 6.0, 6000)


def jonswap(w, Hs, Tp, gamma=GAMMA):
    """JONSWAP wave spectrum S(omega) [m^2 s/rad], scaled so that m0 = Hs^2/16 over W_WIDE."""
    def shape(w):
        wp = 2 * np.pi / Tp
        s = np.where(w <= wp, 0.07, 0.09)
        r = np.exp(-(w - wp) ** 2 / (2 * s**2 * wp**2))
        return w**-5 * np.exp(-1.25 * (wp / w) ** 4) * gamma**r
    scale = (Hs**2 / 16) / np.trapezoid(shape(W_WIDE), W_WIDE)
    return scale * shape(w)


_RAO_CACHE = {}


def force_rao(gap, k):
    key = (gap, k)
    if key not in _RAO_CACHE:
        d = load(gap)
        _, hf = linear_rao(d, k)
        _RAO_CACHE[key] = (d["omega"], hf)
    return _RAO_CACHE[key]


def linear_stats(Hs, Tp, gap, k, duration=1800.0, w=None, hf=None):
    if w is None:
        w, hf = force_rao(gap, k)
    S = jonswap(w, Hs, Tp)
    SF = np.abs(hf) ** 2 * S
    m0, m2 = np.trapezoid(SF, w), np.trapezoid(SF * w**2, w)
    sigma = np.sqrt(m0)
    Tz = 2 * np.pi * np.sqrt(m0 / m2)
    N = duration / Tz
    a = np.sqrt(2 * np.log(N))
    return dict(sigma_F=sigma, Tz=Tz, Emax_F=sigma * (a + 0.5772 / a))


def truncation_fraction(Tp):
    S = jonswap(W_WIDE, 1.0, Tp)
    inside = (W_WIDE >= 0.2) & (W_WIDE <= 2.2)
    return 1 - np.trapezoid(S[inside], W_WIDE[inside]) / np.trapezoid(S, W_WIDE)


def report(name, value, tol, ok):
    print(f"  {name:<60s} {value:>11.4e}   tol {tol:<7g} {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok = True
    print("L1 spectrum normalisation")
    for Hs, Tp in [(1.0, 6), (2.5, 9), (4.0, 12)]:
        m0 = np.trapezoid(jonswap(W_WIDE, Hs, Tp), W_WIDE)
        e = abs(4 * np.sqrt(m0) - Hs) / Hs
        ok &= report(f"Hs={Hs} Tp={Tp}: |4 sqrt(m0) - Hs|/Hs", e, 0.005, e < 0.005)

    print("L2 wave energy outside 0.2-2.2 rad/s (reported; flagged if > 5%)")
    for Tp in [6, 7, 8, 10, 12]:
        f = truncation_fraction(Tp)
        print(f"  Tp = {Tp:>2} s   {100*f:5.2f} %   {'OK' if f < 0.05 else 'FLAG'}")
        ok &= f < 0.05

    print("L3 grid convergence: 60-pt BEM grid vs 600-pt linear interpolation of the complex RAO")
    e3 = []
    for gap in [1.5, 3, 5]:
        for k in [1e5, 1e6, 1e7]:
            w, hf = force_rao(gap, k)
            wf = np.linspace(w[0], w[-1], 600)
            hff = np.interp(wf, w, hf.real) + 1j * np.interp(wf, w, hf.imag)
            for Tp in [6, 9, 12]:
                a = linear_stats(2.0, Tp, gap, k)["sigma_F"]
                b = linear_stats(2.0, Tp, gap, k, w=wf, hf=hff)["sigma_F"]
                e3.append(abs(a - b) / b)
    ok &= report(f"max over {len(e3)} combinations (gap x k x Tp)", max(e3), 0.02, max(e3) < 0.02)

    print("L4 Monte-Carlo: random-phase linear time series, 20 realisations x 3 h, dt 0.2 s")
    rng = np.random.default_rng(20261124)
    for gap, k, Hs, Tp in [(2, 1e7, 2.0, 8), (1.5, 1e5, 3.0, 6), (5, 1e6, 1.5, 11)]:
        w, hf = force_rao(gap, k)
        wf = np.linspace(w[0], w[-1], 2000); dw = wf[1] - wf[0]
        hff = np.interp(wf, w, hf.real) + 1j * np.interp(wf, w, hf.imag)
        amp = np.sqrt(2 * jonswap(wf, Hs, Tp) * dw)
        t = np.arange(0, 3 * 3600, 0.2)
        stds = []
        for _ in range(20):
            ph = rng.uniform(0, 2 * np.pi, wf.size)
            wfj = wf + rng.uniform(-dw / 2, dw / 2, wf.size)   # jitter: avoids the periodic repeat of an equal grid
            Ft = (np.abs(hff) * amp * np.cos(np.outer(t, wfj) + ph + np.angle(hff))).sum(axis=1)
            stds.append(Ft.std())
        s_fd = linear_stats(Hs, Tp, gap, k)["sigma_F"]
        e = abs(np.mean(stds) - s_fd) / s_fd
        ok &= report(f"gap {gap} k {k:.0e} Hs {Hs} Tp {Tp}: sigma_MC vs sigma_FD", e, 0.03, e < 0.03)

    print("L5 zero-stiffness limit")
    r = linear_stats(2.0, 8, 2, 1e-3)["sigma_F"] / linear_stats(2.0, 8, 2, 1e7)["sigma_F"]
    ok &= report("sigma_F(k=1e-3) / sigma_F(k=1e7)", r, 1e-3, r < 1e-3)

    # Figure: sigma_F over (Tp, k) per gap at Hs = 2 m, and linearity in Hs
    Tps = np.linspace(6, 12, 25); ks = np.logspace(5, 7, 25)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for j, gap in enumerate([1.5, 5]):
        Z = np.array([[linear_stats(2.0, Tp, gap, k)["sigma_F"] / 1e3 for Tp in Tps] for k in ks])
        c = ax[j].contourf(Tps, ks, Z, levels=15); ax[j].set_yscale("log")
        ax[j].set(xlabel="Tp [s]", ylabel="k [N/m]", title=f"Linear sigma_F [kN], Hs = 2 m, gap {gap} m")
        fig.colorbar(c, ax=ax[j])
    for k in [1e5, 1e6, 1e7]:
        ax[2].plot(Tps, [linear_stats(2.0, Tp, 2, k)["Emax_F"] / 1e3 for Tp in Tps], label=f"k = {k:.0e}")
    ax[2].set(xlabel="Tp [s]", ylabel="E[max F] in 30 min [kN]", title="Linear expected max, Hs = 2 m, gap 2 m")
    ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "step2_linear_fd.png", dpi=140)
    print("Figure: figures/step2_linear_fd.png")
    print("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)
