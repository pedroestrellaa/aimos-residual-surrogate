"""Step 3 - Synthetic nonlinear truth: Cummins time-domain model of the two connected barges.

    (M + A_inf) x'' + int_0^t K(t - s) x'(s) ds + C x = F_exc(t) - c F_conn(delta) + F_drag(x') + F_gap(delta')

x = [z1, th1, z2, th2]. Built from the SAME step-1 BEM coefficients as the linear model (step 2).
  K(t)    retardation functions, K(t) = (2/pi) int B(w) cos(wt) dw   (B tapered to 0 at 0 and above 2.2 rad/s)
  A_inf   fitted by Ogilvie's relation A(w) = A_inf - (1/w) int K(t) sin(wt) dt  over the BEM band, so that the
          time-domain model reproduces the BEM A(w) and B(w) as closely as the truncated band allows

Three nonlinear terms define the synthetic truth. ⛔ Their coefficients are ASSUMED, not measured:
  N1 quadratic viscous drag      strip-wise on each hull, f(x) = -1/2 rho CD B |w| w,  w(x) = z' - (x - x_i) th'
  N2 hardening connector         F_conn = k delta (1 + (delta / DELTA0)^2)
  N3 quadratic gap damping       F_gap  = -c CG_COEF (g_ref / gap) |delta'| delta'   (on relative motion at the gap)
With all three off, F_conn = k delta and the model must reproduce the linear frequency-domain result (gate G2).

Wave realisation: random phases on 2000 jittered frequencies in 0.2-2.2 rad/s. The same phases also give the
linear frequency-domain force time series (superposition of the step-2 RAO), so linear and nonlinear responses
are compared on the identical sea (common random numbers).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_bem import load, connector_vector, linear_rao  # noqa: E402
from linear_fd import jonswap  # noqa: E402

RHO = 1025.0
# ---- assumed nonlinear coefficients (the synthetic truth). Changing them changes the "truth", not the method.
CD = 1.0            # N1 drag coefficient on the flat bottom (assumed)
DELTA0 = 0.5        # N2 connector hardening length [m] (assumed). Run 1 used 0.2: sigma x10 on soft connectors, not a
                    # correction. Chosen by logs/step3_delta0_scan.log: smallest value with sigma_NL/sigma_L in [0.6, 1.8]
CG_COEF = 5.0e4     # N3 gap damping coefficient at the reference gap [kg/m] (assumed)
G_REF = 1.5         # N3 reference gap [m]
N_STRIPS = 20


@dataclass
class Model:
    gap: float
    dt: float
    Mt: np.ndarray            # M + A_inf
    C: np.ndarray             # hydrostatic stiffness
    Kt: np.ndarray            # retardation kernel samples (n_mem, 4, 4)
    omega: np.ndarray
    F: np.ndarray             # excitation RAO (n_w, 4) complex
    meta: dict = field(default_factory=dict)


HF_SCALE = 1.0      # multiplier on radiation damping above 2.2 rad/s (sensitivity test uses 0.5 and 2.0)
W_KERNEL_MAX = 4.0  # kernel band upper limit [rad/s]


def _damping_spectrum(d, wf, hf_scale):
    """B(omega) on wf in [0, 4] rad/s.
    0-2.2 rad/s: two-body BEM (step 1, validated), linear to 0 below 0.2.
    2.2-4.0 rad/s: block-diagonal single-body BEM with lid, fine mesh (src/bem_high_freq.py) x hf_scale.
    Linear blend on 2.0-2.2 rad/s, cosine taper to 0 on 3.8-4.0 rad/s.
    v1 (dataset quarantined) used only 0-2.2 rad/s with a taper to 0 at 2.2: the connector mode at 2.3-3.5 rad/s
    then had no radiation damping."""
    w, B = d["omega"], d["B"]
    Blo = np.empty((wf.size, 4, 4))
    for i in range(4):
        for j in range(4):
            Blo[:, i, j] = np.interp(wf, np.concatenate([[0.0], w]), np.concatenate([[0.0], B[:, i, j]]))
    hf = np.load(Path(__file__).resolve().parents[1] / "data" / "bem_single_hf.npz")
    Bhi = np.zeros((wf.size, 4, 4))
    for b in range(2):
        for i in range(2):
            for j in range(2):
                Bhi[:, 2 * b + i, 2 * b + j] = np.interp(wf, hf["omega"], hf["B"][:, i, j])
    Bhi *= hf_scale
    a = np.clip((wf - 2.0) / 0.2, 0, 1)[:, None, None]
    Bf = (1 - a) * Blo + a * Bhi
    taper = np.where(wf > W_KERNEL_MAX - 0.2, 0.5 * (1 + np.cos(np.pi * (wf - (W_KERNEL_MAX - 0.2)) / 0.2)), 1.0)
    return Bf * taper[:, None, None]


def build_model(gap, dt=0.05, t_mem=40.0, hf_scale=None):
    d = load(gap)
    w, A = d["omega"], d["A"]
    wf = np.linspace(0.0, W_KERNEL_MAX, 4001)
    Bf = _damping_spectrum(d, wf, HF_SCALE if hf_scale is None else hf_scale)
    t = np.arange(0.0, t_mem, dt)
    # trapezoid rule as a matrix product (same result, ~26 MB instead of a 410 MB broadcast per process)
    wq = np.full(wf.size, wf[1] - wf[0]); wq[[0, -1]] *= 0.5
    Kt = ((2 / np.pi) * (np.cos(np.outer(t, wf)) * wq) @ Bf.reshape(wf.size, 16)).reshape(t.size, 4, 4)
    # Ogilvie: A_inf(w) = A(w) + (1/w) int K sin(wt) dt, evaluated at each BEM frequency, then averaged
    Ainf_w = np.array([A[n] + np.trapezoid(Kt * np.sin(wn * t)[:, None, None], t, axis=0) / wn
                       for n, wn in enumerate(w)])
    band = (w >= 0.4) & (w <= 2.0)
    A_inf = Ainf_w[band].mean(axis=0)
    spread = np.abs(Ainf_w[band] - A_inf).max() / np.abs(A_inf).max()
    meta = dict(A_inf_ogilvie_spread=float(spread),
                A_inf_vs_bem=float(np.abs(A_inf - d["A_inf"]).max() / np.abs(d["A_inf"]).max()),
                K_tail=float(np.abs(Kt[-int(5 / dt):]).max() / np.abs(Kt).max()))
    return Model(gap=gap, dt=dt, Mt=d["M"] + A_inf, C=d["K"], Kt=Kt, omega=w, F=d["F"], meta=meta)


def wave_components(Hs, Tp, seed, n=2000):
    rng = np.random.default_rng(seed)
    w = np.linspace(0.2, 2.2, n); dw = w[1] - w[0]
    w = w + rng.uniform(-dw / 2, dw / 2, n)
    a = np.sqrt(2 * jonswap(w, Hs, Tp) * dw)
    ph = rng.uniform(0, 2 * np.pi, n)
    return w, a, ph


def _superpose(t, w, amp_c, chunk=2000):
    """sum_j Re(amp_c_j exp(-i w_j t)) for complex amplitudes amp_c (n_w, ...), chunked over time.
    exp(-i w t) is Capytaine's convention (and the -i w B in check_bem.linear_rao). Run 1 used exp(+i w t),
    which pairs each excitation with the complex conjugate of its response: gate G2 failed (corr 0.80)."""
    out = np.empty((t.size,) + amp_c.shape[1:])
    for s in range(0, t.size, chunk):
        e = np.exp(-1j * np.outer(t[s:s + chunk], w))
        out[s:s + chunk] = np.real(np.tensordot(e, amp_c, axes=(1, 0)))
    return out


def interp_complex(x, xp, fp):
    return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)


def simulate(model, Hs, Tp, k, seed, duration=1800.0, t_ramp=50.0, t_discard=200.0,
             drag=True, hardening=True, gap_damping=True):
    """Returns dict with the connector force time series of the time-domain model (after discard),
    and the linear frequency-domain superposition force on the identical wave realisation."""
    dt = model.dt
    t = np.arange(0.0, t_discard + duration, dt)
    w, a, ph = wave_components(Hs, Tp, seed)
    cplx = a * np.exp(1j * ph)
    Fj = np.stack([interp_complex(w, model.omega, model.F[:, i]) for i in range(4)], axis=1)
    Fexc = _superpose(t, w, cplx[:, None] * Fj)
    Fexc *= np.clip(t / t_ramp, 0, 1)[:, None]

    L = 20.0
    c = connector_vector(L)
    # strip coordinates relative to each body's centre (pitch about the CoG)
    xs = (np.arange(N_STRIPS) + 0.5) / N_STRIPS * L - L / 2
    dx = L / N_STRIPS
    qd = 0.5 * RHO * CD * 10.0 * dx                      # B = 10 m
    cg = CG_COEF * (G_REF / model.gap)
    Minv = np.linalg.inv(model.Mt)
    nmem = model.Kt.shape[0]
    Kflip = model.Kt[::-1]                               # so that history window aligns with kernel
    vhist = np.zeros((t.size + nmem, 4))                 # zero-padded velocity history

    def conn(delta):
        return k * delta * (1 + (delta / DELTA0) ** 2) if hardening else k * delta

    def rhs(x, v, fe, mu):
        delta, ddelta = c @ x, c @ v
        f = fe - mu - model.C @ x - c * conn(delta)
        if gap_damping:
            f -= c * cg * abs(ddelta) * ddelta
        if drag:
            for b in range(2):
                wv = v[2 * b] - xs * v[2 * b + 1]
                fs = -qd * np.abs(wv) * wv
                f[2 * b] += fs.sum()
                f[2 * b + 1] += -(xs * fs).sum()
        return v, Minv @ f

    x = np.zeros(4); v = np.zeros(4)
    xs_out = np.empty((t.size, 4))
    for n in range(t.size):
        vhist[n + nmem - 1] = v
        hist = vhist[n:n + nmem]                         # velocities from t_n - (nmem-1) dt ... t_n
        mu = np.einsum("kij,kj->i", Kflip, hist) * dt    # rectangle rule on the memory integral
        xs_out[n] = x
        fe0 = Fexc[n]; fe1 = Fexc[min(n + 1, t.size - 1)]; feh = 0.5 * (fe0 + fe1)
        k1x, k1v = rhs(x, v, fe0, mu)
        k2x, k2v = rhs(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, feh, mu)
        k3x, k3v = rhs(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, feh, mu)
        k4x, k4v = rhs(x + dt * k3x, v + dt * k3v, fe1, mu)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

    keep = t >= t_discard
    delta = xs_out[keep] @ c
    F_td = conn(delta)
    # linear FD superposition on the identical realisation (no ramp needed: steady state by construction)
    _, hf = linear_rao(load(model.gap), k)
    hfj = interp_complex(w, model.omega, hf)
    F_fd = _superpose(t[keep], w, cplx * hfj)
    return dict(t=t[keep], F_td=F_td, F_fd=F_fd, x=xs_out[keep])
