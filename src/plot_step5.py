"""Learning curves, run A (pre-registered) and run B (post-hoc early stopping). Mean ± sd over 5 repeats."""
import csv, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; fl = json.load(open(ROOT / "results" / "step5_noise_floor.json"))
NS = [25, 50, 100, 200, 400, 600]
STY = {"PF": ("k", "--"), "D-raw": ("tab:gray", "-"), "D-pf": ("tab:blue", "-"), "RES": ("tab:red", "-")}
LAB = {"PF": "potential flow only", "D-raw": "direct, raw inputs", "D-pf": "direct + PF features", "RES": "residual on PF"}
fig, ax = plt.subplots(2, 4, figsize=(16, 7), sharex=True)
for row, run in enumerate("AB"):
    R = list(csv.DictReader(open(ROOT / "results" / f"step5{run}_results.csv")))
    for col, (t, split) in enumerate([("peak", "rmse_in"), ("peak", "rmse_gap3"), ("sigma", "rmse_in"), ("sigma", "rmse_gap3")]):
        a = ax[row, col]
        for m in ["PF", "D-raw", "D-pf", "RES"]:
            mu = [np.mean([float(r[split]) for r in R if r["target"] == t and r["model"] == m and int(r["N"]) == N]) for N in NS]
            sd = [np.std([float(r[split]) for r in R if r["target"] == t and r["model"] == m and int(r["N"]) == N]) for N in NS]
            c, ls = STY[m]; a.errorbar(NS, mu, yerr=sd, color=c, ls=ls, marker="o", ms=3, capsize=2, label=LAB[m])
        a.axhline(fl[t], color="green", lw=0.8, ls=":", label="noise floor")
        a.set_xscale("log"); a.minorticks_off(); a.set_xticks(NS); a.set_xticklabels(NS)
        a.set_title(f"run {run}{' (pre-registered)' if run == 'A' else ' (post-hoc: early stopping)'}\n"
                    f"{'30-min peak' if t == 'peak' else 'std'} - {'in-distribution' if split == 'rmse_in' else 'unseen gap 3 m'}", fontsize=9)
        if col in (0, 2): a.set_ylabel("RMSE of ln(load)")
        if row == 1: a.set_xlabel("training cases N")
ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "figures" / "step5_learning_curves.png", dpi=130)
print("Figure: figures/step5_learning_curves.png")
