"""Figures for the README and the report: setup schematic and headline result (run B, in-distribution)."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"


def schematic():
    fig, ax = plt.subplots(figsize=(10, 2.9))
    x = np.linspace(-27, 25, 900)
    eta = 0.35 * np.sin(0.55 * x)
    inside = ((x > -21) & (x < -1)) | ((x > 1) & (x < 21))
    ax.fill_between(x, eta, -4.0, color="tab:blue", alpha=0.08, zorder=0)
    ax.plot(np.where(inside, np.nan, x), eta, color="tab:blue", lw=1.2, zorder=1)
    for x0, lab in [(-21, "body 1"), (1, "body 2")]:
        ax.add_patch(Rectangle((x0, -2.0), 20, 2.9, fc="0.85", ec="k", lw=1.2, zorder=2))
        ax.text(x0 + 10, -0.55, lab, ha="center", va="center", fontsize=10, zorder=3)
    ax.plot(np.linspace(-1, 1, 9), 0.45 + 0.25 * np.array([0, 1, -1, 1, -1, 1, -1, 1, 0]), color="tab:red", lw=1.6, zorder=3)
    ax.text(0, 2.55, r"connector  $F_c = k\,\delta\,(1+(\delta/\delta_0)^2)$", ha="center", color="tab:red", fontsize=10)
    ax.plot([0, 0], [0.8, 2.4], color="tab:red", lw=0.6)
    ax.annotate("", xy=(-1, -2.5), xytext=(1, -2.5), arrowprops=dict(arrowstyle="<->", lw=1))
    ax.text(0, -3.45, "gap $g$", ha="center", fontsize=10)
    ax.text(0, -1.5, "gap\ndamping", ha="center", fontsize=7, color="tab:purple", zorder=3)
    for x0, zl, tl, dx in [(-11, "$z_1$", r"$\theta_1$", -6), (11, "$z_2$", r"$\theta_2$", 4)]:
        ax.add_artist(FancyArrowPatch((x0, 0.9), (x0, 2.3), arrowstyle="->", mutation_scale=12))
        ax.text(x0 + 0.5, 1.9, zl, fontsize=11)
        ax.add_artist(FancyArrowPatch((x0 + dx, 1.3), (x0 + dx + 2, 1.3), connectionstyle="arc3,rad=-0.5", arrowstyle="->", mutation_scale=10))
        ax.text(x0 + dx + 0.6, 1.9, tl, fontsize=11)
    ax.add_artist(FancyArrowPatch((-26.5, 1.4), (-22, 1.4), arrowstyle="->", mutation_scale=14, color="tab:blue"))
    ax.text(-26.5, 1.8, "head seas,\nJONSWAP", fontsize=8, color="tab:blue")
    ax.text(-11, -3.3, "quadratic drag on the bottom", ha="center", fontsize=8, color="0.3")
    ax.text(11, -3.3, r"each body $20 \times 10 \times 2$ m", ha="center", fontsize=9)
    ax.set_xlim(-27, 25); ax.set_ylim(-3.9, 3.1); ax.set_aspect("equal"); ax.axis("off")
    fig.savefig(FIG / "setup_schematic.png", dpi=150, bbox_inches="tight", pad_inches=0.05)


def headline():
    R = list(csv.DictReader(open(ROOT / "results" / "step5B_results.csv")))
    fl = json.load(open(ROOT / "results" / "step5_noise_floor.json"))
    NS = [25, 50, 100, 200, 400, 600]
    sty = {"PF": ("k", "--", "potential flow only (no learning)"), "D-raw": ("tab:gray", "-", "direct, raw inputs"),
           "D-pf": ("tab:blue", "-", "direct + potential-flow features"), "RES": ("tab:red", "-", "residual on potential flow")}
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (t, ttl) in zip(ax, [("peak", "30-min peak connector force"), ("sigma", "Standard deviation of connector force")]):
        for m, (c, ls, lab) in sty.items():
            v = [[float(r["rmse_in"]) for r in R if r["target"] == t and r["model"] == m and int(r["N"]) == N] for N in NS]
            a.errorbar(NS, [np.mean(x) for x in v], yerr=[np.std(x) for x in v], color=c, ls=ls, marker="o", ms=3,
                       capsize=2, label=lab)
        a.axhline(fl[t], color="green", ls=":", lw=1, label="noise floor of a 30-min record")
        a.set_xscale("log"); a.minorticks_off(); a.set_xticks(NS); a.set_xticklabels(NS)
        a.set(xlabel="training cases N", ylabel="RMSE of ln(load)  (about relative error)", title=ttl)
    ax[0].legend(fontsize=8)
    fig.suptitle("Test error vs training budget (early stopping, 5 repeats, mean ± sd)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "headline_result.png", dpi=150)


if __name__ == "__main__":
    schematic(); headline()
    print("figures/setup_schematic.png, figures/headline_result.png")
