"""Diagnostic for the step in the dataset residual at log10 k ~ 5.8: natural frequencies of the coupled system
(M + A(omega)) x'' + (C + k c c^T) x = 0, solved with A at the mode's own frequency (fixed-point), gap 3 m."""
import sys; from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_bem import load, connector_vector
d = load(3); c = connector_vector(20.0); w = d["omega"]
for lk in [5.0, 5.4, 5.6, 5.7, 5.8, 5.9, 6.0, 6.5, 7.0]:
    k = 10**lk; Kt = d["K"] + k * np.outer(c, c); out = []
    for guess in [0.8, 1.3, 2.0, 3.0]:
        wn = guess
        for _ in range(50):
            A = np.array([np.interp(min(wn, w[-1]), w, d["A"][:, i, j]) for i in range(4) for j in range(4)]).reshape(4, 4)
            ev, vec = np.linalg.eig(np.linalg.solve(d["M"] + A, Kt))
            ws = np.sqrt(np.real(ev)); i = np.argmin(abs(ws - wn)); wn = 0.5 * wn + 0.5 * ws[i]
        v = np.real(vec[:, i]); out.append((wn, abs(c @ v) / np.linalg.norm(v)))
    modes = sorted(set((round(a, 2), round(b, 2)) for a, b in out))
    print(f"log10 k {lk:<4}: modes (omega rad/s, |connector stretch| share): {modes}")
