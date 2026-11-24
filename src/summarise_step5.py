"""Summary table of a step-5 results file. Usage: python src/summarise_step5.py results/step5A_results.csv"""
import csv, json, sys, numpy as np
R = list(csv.DictReader(open(sys.argv[1]))); fl = json.load(open("results/step5_noise_floor.json"))
for t in ["peak", "sigma"]:
    print(f"\n=== target {t}  (noise floor RMSE ln = {fl[t]:.4f}) ===  mean ± sd over 5 repeats")
    print(f"{'model':<6}" + "".join(f"{'N='+str(N):>22}" for N in [25, 50, 100, 200, 400, 600]))
    for split in ["rmse_in", "rmse_gap3"]:
        print(f"  [{split}]")
        for m in ["PF", "D-raw", "D-pf", "RES"]:
            row = f"{m:<6}"
            for N in [25, 50, 100, 200, 400, 600]:
                v = [float(r[split]) for r in R if r["target"] == t and r["model"] == m and int(r["N"]) == N]
                row += f"{np.mean(v):>14.4f} ±{np.std(v):.4f}"
            print(row)
    print("  conformal (N=600, 90% target): coverage in / gap3, half-width (ln)")
    for m in ["PF", "D-raw", "D-pf", "RES"]:
        v = [r for r in R if r["target"] == t and r["model"] == m and int(r["N"]) == 600]
        print(f"  {m:<6} cov_in {np.mean([float(x['cov_in']) for x in v]):.3f}  cov_gap3 {np.mean([float(x['cov_gap3']) for x in v]):.3f}  halfwidth {np.mean([float(x['q_halfwidth_ln']) for x in v]):.4f}")
