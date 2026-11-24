"""Step 5 - Direct vs residual learning of connector loads, at equal training budget (PyTorch).

EVERYTHING BELOW WAS FIXED BEFORE THE FIRST TRAINING RUN (2026-11-24). No hyperparameter was tuned on test data.

Splits (data/dataset.csv, 1200 cases, split seed 7):
  test_gap3   all 240 cases with gap = 3 m. Never seen in training or calibration (unseen configuration).
  test_in     20% of the other 960 cases (192), in-distribution test.
  calib       168 of the remaining 768, used only for split-conformal intervals.
  train pool  the last 600. Training sets are nested random subsets, N in {25, 50, 100, 200, 400, 600}, 5 repeats each.

Targets: y = ln(max_NL) (30-min peak connector force, main) and y = ln(sigma_NL) (secondary).

Models (same MLP, same optimiser, same epochs; only the inputs and the output parameterisation differ):
  PF    potential flow alone, no learning: y_hat = ln(Emax_L) or ln(sigma_L)
  D-raw direct, raw inputs [Hs, Tp, gap, log10k]                               y_hat = f(x)
  D-pf  direct, raw + potential-flow features [ln sigma_L, ln Tz_L, ln Emax_L]  y_hat = f(x)
  RES   residual on the potential-flow prediction, same inputs as D-pf          y_hat = y_PF + g(x)
MLP: inputs standardised on the training set, 2 hidden layers x 64, SiLU, Adam lr 1e-3, weight decay 1e-4,
full batch, 2000 epochs. Output layer: RES initialised to zero (starts at potential flow); direct models get
the output bias set to the training-target mean.

Metrics: RMSE of ln y (≈ relative error) on test_in and test_gap3.
Noise floor: a 30-min statistic is random even with exact physics. Estimated by how well the exact linear
expectation predicts the linear statistic of the same record: RMSE ln(max_Lr / Emax_L), RMSE ln(sigma_Lr / sigma_L).
Run B (POST-HOC, designed after run A showed overfitting at small N; see LOG): --protocol B adds early stopping.
Each training set is split 80/20 (>= 5 validation cases, split seed = repeat); validation loss every 10 epochs;
the best-validation weights are kept. Identical for D-raw, D-pf and RES. Output: results/step5B_results.csv.
Conformal: split-conformal, 90% target, absolute residual score on calib, coverage and width on both tests (N = 600).
"""
import csv
import json
import sys
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NS = [25, 50, 100, 200, 400, 600]
REPEATS = 5
EPOCHS, LR, WD, HIDDEN = 2000, 1e-3, 1e-4, 64
TARGETS = {"peak": ("max_NL", "Emax_L"), "sigma": ("sigma_NL", "sigma_L")}


def load_splits():
    d = np.genfromtxt(ROOT / "data" / "dataset.csv", delimiter=",", names=True)
    rng = np.random.default_rng(7)
    gap3 = np.where(d["gap"] == 3.0)[0]
    rest = rng.permutation(np.where(d["gap"] != 3.0)[0])
    test_in, calib, pool = rest[:192], rest[192:360], rest[360:]
    assert len(pool) == 600 and len(set(pool) | set(calib) | set(test_in) | set(gap3)) == 1200
    return d, dict(test_gap3=gap3, test_in=test_in, calib=calib, pool=pool)


def features(d, idx, kind):
    raw = np.column_stack([d["Hs"][idx], d["Tp"][idx], d["gap"][idx], d["log10k"][idx]])
    if kind == "D-raw":
        return raw
    pf = np.column_stack([np.log(d["sigma_L"][idx]), np.log(d["Tz_L"][idx]), np.log(d["Emax_L"][idx])])
    return np.column_stack([raw, pf])


class MLP(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, HIDDEN), torch.nn.SiLU(),
                                       torch.nn.Linear(HIDDEN, HIDDEN), torch.nn.SiLU(),
                                       torch.nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


PROTOCOL = "A"


def fit_predict(Xtr, ytr, base_tr, Xev_list, base_ev_list, residual, seed):
    torch.manual_seed(seed)
    if PROTOCOL == "B":
        rs = np.random.default_rng(seed); perm = rs.permutation(len(ytr)); nv = max(5, int(0.2 * len(ytr)))
        va, trn = perm[:nv], perm[nv:]
        Xva, yva, bva = Xtr[va], ytr[va], base_tr[va]
        Xtr, ytr, base_tr = Xtr[trn], ytr[trn], base_tr[trn]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    xtr = T((Xtr - mu) / sd)
    target = T(ytr - base_tr) if residual else T(ytr)
    m = MLP(Xtr.shape[1])
    last = m.net[-1]
    with torch.no_grad():
        if residual:
            last.weight.zero_(); last.bias.zero_()
        else:
            last.bias.fill_(float(target.mean()))
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    if PROTOCOL == "B":
        xva = T((Xva - mu) / sd); tva = T(yva - bva) if residual else T(yva)
        best, best_state = np.inf, None
    for ep in range(EPOCHS):
        opt.zero_grad()
        loss = torch.mean((m(xtr) - target) ** 2)
        loss.backward(); opt.step()
        if PROTOCOL == "B" and ep % 10 == 0:
            with torch.no_grad():
                lv = float(torch.mean((m(xva) - tva) ** 2))
            if lv < best:
                best, best_state = lv, {k: v.clone() for k, v in m.state_dict().items()}
    if PROTOCOL == "B":
        m.load_state_dict(best_state)
    out = []
    with torch.no_grad():
        for Xev, bev in zip(Xev_list, base_ev_list):
            p = m(T((Xev - mu) / sd)).numpy().astype(float)
            out.append(p + bev if residual else p)
    return out


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def job(args):
    global PROTOCOL
    torch.set_num_threads(1)
    PROTOCOL = args[4] if len(args) > 4 else "A"
    args = args[:4]
    target, model, N, rep = args
    d, S = load_splits()
    ycol, pfcol = TARGETS[target]
    rng = np.random.default_rng(1000 + rep)
    tr = rng.permutation(S["pool"])[:N]          # nested within a repeat: same permutation, first N
    ev_sets = ["test_in", "test_gap3", "calib"]
    y = lambda idx: np.log(d[ycol][idx])
    base = lambda idx: np.log(d[pfcol][idx])
    if model == "PF":
        preds = [base(S[s]) for s in ev_sets]
    else:
        preds = fit_predict(features(d, tr, model), y(tr), base(tr),
                            [features(d, S[s], model) for s in ev_sets], [base(S[s]) for s in ev_sets],
                            residual=(model == "RES"), seed=rep)
    res = dict(target=target, model=model, N=N, rep=rep,
               rmse_in=rmse(preds[0], y(S["test_in"])), rmse_gap3=rmse(preds[1], y(S["test_gap3"])))
    if N == NS[-1]:
        q_level = np.ceil(0.9 * (len(S["calib"]) + 1)) / len(S["calib"])
        q = float(np.quantile(np.abs(preds[2] - y(S["calib"])), q_level, method="higher"))
        for name, p, s in [("in", preds[0], "test_in"), ("gap3", preds[1], "test_gap3")]:
            res[f"cov_{name}"] = float(np.mean(np.abs(p - y(S[s])) <= q))
        res["q_halfwidth_ln"] = q
    return res


if __name__ == "__main__":
    PROTOCOL = sys.argv[sys.argv.index("--protocol") + 1] if "--protocol" in sys.argv else "A"
    t0 = time.time()
    d, S = load_splits()
    floor = {"peak": rmse(np.log(d["max_Lr"]), np.log(d["Emax_L"])),
             "sigma": rmse(np.log(d["sigma_Lr"]), np.log(d["sigma_L"]))}
    print("Noise floor (RMSE ln, exact linear expectation vs same-record linear statistic):",
          {k: round(v, 4) for k, v in floor.items()}, flush=True)
    jobs = [(t, "PF", N, 0) for t in TARGETS for N in NS] + \
           [(t, m, N, r) for t, m, N, r in product(TARGETS, ["D-raw", "D-pf", "RES"], NS, range(REPEATS))]
    jobs = [j + (PROTOCOL,) for j in jobs]
    with Pool(30) as p:
        results = p.map(job, jobs)
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    keys = sorted({k for r in results for k in r}, key=lambda k: list(results[0]).index(k) if k in results[0] else 99)
    with open(out / f"step5{PROTOCOL}_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(results)
    json.dump(floor, open(out / "step5_noise_floor.json", "w"))
    print(f"protocol {PROTOCOL}: {len(results)} runs in {time.time()-t0:.0f} s -> results/step5{PROTOCOL}_results.csv", flush=True)
