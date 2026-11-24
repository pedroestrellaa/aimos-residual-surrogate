"""Diagnostic (v2 after the fix: kernel damping now validated to 4.0 rad/s; criterion moved to that edge): share of the truth connector-force variance above the BEM band (omega > 2.2 rad/s), where the
time-domain model has no radiation damping. If that share is material, the soft-connector residual is partly an
artifact of the truncated kernel, not physics. Criterion fixed before running: share < 2% of variance."""
import sys; from pathlib import Path; from multiprocessing import Pool
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cummins_td import build_model, simulate

def run(a):
    lk, Hs, Tp = a
    r = simulate(build_model(3.0), Hs, Tp, 10**lk, seed=5)
    F = r["F_td"] - r["F_td"].mean(); dt = r["t"][1] - r["t"][0]
    P = np.abs(np.fft.rfft(F))**2; w = 2*np.pi*np.fft.rfftfreq(F.size, dt)
    tot = P[1:].sum()
    share_mid = P[(w > 2.2) & (w <= 4.0)].sum() / tot      # damped by the extended kernel: model physics (springing-like)
    share = P[w > 4.0].sum() / tot                          # beyond the validated kernel band: must be < 2%
    peak = w[w > 2.2][np.argmax(P[w > 2.2])]
    return a, share, share_mid, peak, np.log(r["F_td"].std() / r["F_fd"].std())

if __name__ == "__main__":
    cases = [(lk, Hs, Tp) for lk in [5.0, 5.3, 5.5, 5.7, 5.9, 6.2, 7.0] for Hs, Tp in [(1.0, 9.0), (3.5, 7.0)]]
    with Pool(14) as p: res = p.map(run, cases)
    ok = True
    for (lk, Hs, Tp), s, sm, pk, r in res:
        ok &= s < 0.02
        print(f"log10k {lk:<4} Hs {Hs} Tp {Tp}: share 2.2-4 rad/s {100*sm:6.2f} %  share > 4 rad/s {100*s:5.2f} %  (peak > 2.2 at {pk:.2f})   r_sigma {r:+.3f}   {'OK' if s < 0.02 else 'ARTIFACT RISK'}")
    print("G5 out-of-band:", "PASS" if ok else "FAIL")
