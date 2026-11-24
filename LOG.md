# Build log: provenance of every step

Each entry records what was done, the exact command, what came out, and what was decided. Failed runs are
kept under `logs/` with their original output, not deleted. Times are local (UTC-3, Recife).

Environment: WSL2 Ubuntu, Python 3.12.3, venv `~/.venvs/aimos`. Package versions are pinned in `requirements.lock`.

---

## 2026-11-24: Step 0, environment

- Wheels downloaded with `pip download`, without installing. torch came from the official PyTorch CPU index
  (`download.pytorch.org/whl/cpu`), everything else from PyPI.
- The SHA256 of every wheel was compared with the hash its index publishes. **30/30 match.** Record:
  `provenance_wheels.tsv` (file, local SHA256, source, result).
- Installed offline from the verified files only: `pip install --no-index --find-links wheels ...`
- Result: torch 2.15.0+cpu, capytaine 3.0.0, scipy 1.18.1, numpy 2.5.5.

## 2026-11-24: Step 1, linear potential-flow coefficients (Capytaine BEM)

**Setup.** Two identical box barges, L × B × T = 20 × 10 × 2 m, end to end along x, gap g, head seas.
DOFs: heave and pitch per body (4 in total), solved as one Multibody so that hydrodynamic interaction is included.
Pitch radius of gyration 0.25 L (assumed). Panel size 1.0 m (640 panels), 60 frequencies.
Script: `src/bem_two_barges.py`. Checks: `src/check_bem.py`.

**Checks** (tolerances fixed before the runs; none was loosened afterwards):
C1 reciprocity A_ij = A_ji, B_ij = B_ji (1%) · C2 identical bodies give identical radiation diagonals (1%) ·
C3 long-wave limit |F_heave|/K33 → 1 (5%) · C4 mesh convergence 1.0 m vs 0.5 m panels (5%) · C5 B_ii ≥ 0.

| Run | Change | Result | Log |
|---|---|---|---|
| 1 | ω 0.2–2.5 rad/s, no lid, gaps 0.5–5 m | **FAIL** C1, C5 at ω ≥ 2.38 rad/s, the first irregular frequency (Capytaine estimate 2.385 rad/s) | `step1_*_RUN1_*` |
| 2 | Interior lid via `mesh.generate_lid()` | **FAIL**, with numbers identical to run 1. The lid came back empty and was silently ignored. | `step1_*_RUN2_*` |
| 3 | Lid built explicitly as a waterplane rectangle | C5 fixed, but **C4 FAIL**: B33 differs by 11% between meshes. Diagnostic `src/diag_lid_test.py`: the lid changes B33 by 17% at ω = 0.8 rad/s, far from any irregular frequency. The lid corrupts results here. | `step1_*_RUN3_*` |
| 4 | No lid, ω capped at 2.2 rad/s; gap 0.5 m dropped (panel 1 m > gap) | All pass except gap 1 m: C1 = 1.02% > 1% (panel = gap) | `step1_checks_RUN4_*` |
| 4b | Gap 1 m dropped as well | **ALL CHECKS PASS** | `step1_bem_run.log`, `step1_checks.log` |

**Final dataset:** gaps 1.5, 2, 3, 4, 5 m at 1.0 m panels, plus gap 2 m at 0.5 m panels for convergence.
Max reciprocity error 0.49%. Mesh convergence ≤ 0.7% on A33, B33 and |F3|. Long-wave limit 0.959.

**Observations** (`figures/step1_bem_checks.png`):
- The heave-heave interaction added mass between the two bodies falls from 0.30 m to 0.25 m as the gap opens from 1.5 to 5 m.
- The linear connector force RAO peaks near heave resonance (ω ≈ 1.25–1.35 rad/s). Above k ≈ 1e6 N/m the
  connector acts as rigid, so the informative stiffness range is 1e5–1e7 N/m.
- A sharp feature at ω ≈ 1.85–1.95 rad/s for small gaps is the potential-flow gap resonance. This is the effect
  known to be overpredicted by potential flow, because there is no viscous damping in the gap.

**Limitations carried forward:**
(1) ω ≤ 2.2 rad/s. Sea states are restricted to Tp ≥ 6 s so that the truncated tail stays small, and the linear
baseline and the synthetic truth use the same truncation.
(2) Gaps below 1.5 m would need panels finer than the gap.
(3) Mass distribution assumed (CoG at the waterline, k_yy = 0.25 L).

## 2026-11-24: Step 2, linear frequency-domain connector load (the potential-flow prediction)

Script: `src/linear_fd.py`. For each case (Hs, Tp, gap, k), the step-1 connector RAO is combined with a JONSWAP spectrum
(γ = 3.3). The script returns σ_F, the zero-upcrossing period of the force Tz, and the Rayleigh expected maximum over 30 min.
Run: `python src/linear_fd.py` → `logs/step2_checks.log`.

| Check | Result |
|---|---|
| L1 spectrum normalisation, 4√m0 = Hs | error ≤ 2e-16 ✅ |
| L2 wave energy outside 0.2–2.2 rad/s | Tp 6 s: 4.0% · 7 s: 2.2% · 8 s: 1.3% · 10 s: 0.5% · 12 s: 0.3%, all < 5% ✅ |
| L3 grid convergence, 60-pt BEM grid vs 600-pt interpolation (27 cases) | max 0.10% ✅ |
| L4 independent Monte-Carlo, 20 × 3 h random-phase series, 3 cases | σ within 0.14–0.26% ✅ |
| L5 zero-stiffness limit | ratio 1.5e-8 ✅ |

Run 1 passed, but L3 only printed when a case failed: silence was the evidence. It was rewritten to print the
maximum error and rerun (`logs/step2_checks_RUN1_L3_silent.log` kept).

**Observations** (`figures/step2_linear_fd.png`): σ_F falls with Tp, as the spectrum moves away from heave resonance
(4.6 s). It rises with k and saturates above about 1e6 N/m. It is larger at the 5 m gap than at the 1.5 m gap
(Hs = 2 m: up to 85 vs 68 kN). The linear model is exactly proportional to Hs: any Hs dependence beyond that must come from the
nonlinear truth (step 3), which is what makes the residual informative.

**Decision:** sea-state range for the dataset is Tp ∈ [6, 12] s (truncation ≤ 4%).

## 2026-11-24: Step 3, synthetic nonlinear truth (Cummins time domain)

Script: `src/cummins_td.py`. Checks: `src/check_td.py` → `logs/step3_checks.log`.
The model is built from the **same** step-1 BEM coefficients. Retardation kernel K(t) = (2/π)∫B cos ωt dω, with B tapered
to zero at 0 and above 2.2 rad/s. A∞ comes from Ogilvie's relation. Time integration is RK4 at dt = 0.05 s, with the
memory integral held constant within each step. Each case runs 1800 s, after a 200 s discard and a 50 s ramp.
Linear and nonlinear responses are computed on the **identical wave realisation** (common random numbers).

**Nonlinear terms (the synthetic truth). ⛔ The coefficients are assumed, not measured:**
N1 strip-wise quadratic drag on the flat bottom, CD = 1.0 · N2 hardening connector F = kδ(1 + (δ/0.5 m)²) ·
N3 quadratic damping on relative velocity at the gap, 5e4 kg/m × (1.5 m / gap).

| Run | What | Result | Log |
|---|---|---|---|
| smoke 1 | linear TD vs FD, gap 2 m | **FAIL**: σ −17%, correlation 0.80. Time series were built with e^{+iωt}, but Capytaine and the step-2 equation use e^{−iωt}, so each excitation was paired with the conjugate of its response. Step 2's Monte-Carlo check used \|H\| only and could not see the phase. | `step3_smoke_RUN1_*` |
| smoke 2 | sign fixed | σ −0.8%, correlation 0.99995 | `step3_smoke_RUN2_*` |
| checks 1 | full gate, DELTA0 = 0.2 m | gates pass, but G4 shows the hardening term multiplying σ ×10 on soft connectors. That truth would replace potential flow, not correct it. | `step3_checks_RUN1_*` |
| scan | DELTA0 ∈ {0.5, 0.75, 1.0, 1.5}, Hs 4 m corners. Criterion **fixed before the scan**: σ_NL/σ_L ∈ [0.6, 1.8]; take the smallest DELTA0 that meets it | 0.5 m: σ ratio 0.91–1.28, max ratio 0.89–2.17 → **DELTA0 = 0.5 m** | `step3_delta0_scan.log` |
| checks 2 | final | **ALL CHECKS PASS** | `step3_checks.log` |

Final gate results: G1 kernel (5 gaps): Ogilvie spread ≤ 1.7%, A∞ vs BEM ≤ 0.7%, kernel tail ≤ 0.8%.
**G2 linear gate (12 cases, terms off): |σ_TD/σ_FD − 1| ≤ 1.3%, time-series RMS difference ≤ 2.2%.**
G3 dt 0.05 vs 0.025 s (terms on): σ ≤ 0.5%, max ≤ 0.4%.

**G4 effect of each term** (Hs 3 m, Tp 7 s), σ_NL/σ_L:
drag 0.88–0.95 · hardening 0.99 (stiff) to 1.46 (soft) · gap damping 0.84 (soft, gap 1.5 m) to 0.99 (stiff) ·
all on 0.94–1.22, peaks up to 1.74. The residual changes sign across the design space: it is a correction
that depends on the case, not a constant factor. Stiff connectors see mainly drag. Soft ones see hardening, which is
partly offset by gap damping, more strongly at small gaps.

Figure `figures/step3_time_domain.png`: top panel, the linear gate (curves coincide); bottom panel, the truth with sharper peaks.

**What this truth is and is not:** it is a controlled, reproducible nonlinear model with the right qualitative
physics (viscous damping, nonlinear connector, damped gap motion). It is **not** CFD-FEM, and its coefficient values
are chosen, not measured. The mini-project tests the *method* (residual vs direct learning), not the true magnitude of
real nonlinear loads.

## 2026-11-24: Step 4, dataset v1, generated and then quarantined

Script: `src/make_dataset.py`. 1,200 cases, Latin hypercube over Hs ∈ [0.5, 4], Tp ∈ [6, 12], log10 k ∈ [5, 7], gap cycling
through {1.5, 2, 3, 4, 5}; wave seed 10000 + i. Determinism: case 7 run twice gives bit-identical results ✅.
Run time was 60 min against about 9 min expected. One case takes 11.7 s alone; 28 workers each running multi-threaded
BLAS oversubscribed the cores. Next runs set `OMP_NUM_THREADS=1`.
Checks (`logs/step4_checks_v1.log`): all pass. Mean σ_Lr/σ_L − 1 = 0.0002; coverage ≥ 120 per decile; 240 per gap.
ln σ_Lr explains 99.66% of the variance of ln σ_NL. The residual in σ is −6% to +6%, and in the 30-min maximum −10% to +57%.

**Problem found** (`figures/step4_dataset_v1.png`): a sharp step in the residual at log10 k ≈ 5.8.
- `src/diag_connector_mode.py`: the connector (relative) mode moves from 1.84 rad/s (k = 1e5) to 2.95 rad/s (10^5.8),
  out of the BEM band.
- `src/diag_out_of_band.py`, criterion fixed in advance (< 2% of variance above 2.2 rad/s): soft connectors in high seas
  put **4–11% of the truth's variance at 2.3–3.5 rad/s**. In that range the time-domain kernel has no radiation
  damping (B tapered to 0 above 2.2). The soft-connector residual is therefore **partly a numerical artifact**.
  The linear gate G2 could not detect it, because a linear run puts no energy outside the wave band.
- **Dataset v1 is quarantined** as `data/dataset_v1_INVALID_soft_connectors.csv`. Nothing has been trained on it.

**Side result, which settles the open step-1 question** (`src/diag_haskind.py`, `logs/step4_diag_haskind.log`): the Haskind relation
(damping from the excitation force over 72 headings, i.e. energy conservation) against directly computed B33, single box:
no lid, ratio 1.035–1.11 for ω 0.8–2.8 rad/s, failing only at 3.5 rad/s · lid at −0.1 m, ratio 0.52–2.0.
**The no-lid solution is the consistent one;** the lid as I set it up was wrong. No-lid BEM is also usable above 2.2 rad/s
away from the narrow irregular-frequency bands.

## 2026-11-24: Step 4 fix, radiation damping up to 4 rad/s

**Root cause of every lid failure since step 1:** Capytaine 3.0's `mesh_rectangle(size=(a, b))` puts `a` along **y** and
`b` along x. Capytaine's own source carries the comment *"TODO Fix: Exchange x and y in mesh_rectangle"*.
My "20 × 10 m" lid was 10 × 20 m: rotated 90°, reaching 5 m beyond each side of the hull.
- `src/diag_lid_recipe.py`: Capytaine's `generate_lid` works on a cylinder (B33 change ≤ 1.2%) and returns nothing for the box.
- `src/diag_lid_box.py`: hand-built lid, 12 variants, all off by 15–20%. With the axes swapped back: ≤ 0.7% ✅.
- Haskind with the corrected lid: 1.0 m panels still fail above 2.4 rad/s (`step4fix_haskind_correct_lid.log`). With 0.5 m panels,
  damping is positive and monotonic to 3.6 rad/s, and the radiation/Haskind ratio is 1.06 at 2.2 and 1.2–1.3 at 2.6–3.2 rad/s (`step4fix_haskind_fine.log`).

**Fix** (`src/bem_high_freq.py`, `cummins_td._damping_spectrum`): the time-domain damping spectrum now covers 0–4 rad/s.
Up to 2.2 rad/s it is the validated two-body BEM (step 1). Above 2.2 rad/s it is the single-body BEM, 0.5 m panels plus the
corrected lid, block-diagonal, so interaction between the bodies is neglected (waves under 13 m long). The two are blended
linearly over 2.0–2.2 rad/s; the join differs by 13% in heave and 21% in pitch. Cosine taper over 3.8–4.0 rad/s.
Two-body fine-mesh BEM with lids was not run: it would take hours.
The kernel integral was rewritten as a matrix product; it agrees with the old form to 4.5e-15
(`step4fix_kernel_rewrite_equivalence.log`), and it removes a 410 MB array per worker, which had made a check run take
over an hour (stopped by PH; G1–G3 had completed).

**Re-validation:**
| Check | Result | Log |
|---|---|---|
| G1 kernel, 5 gaps | Ogilvie spread ≤ 1.2% (was 1.7%), A∞ vs BEM ≤ 0.7%, tail ≤ 0.7% ✅ | `step4fix_td_checks.log` |
| G2 linear gate, 12 cases | σ ≤ 1.5%, RMS ≤ 1.6% ✅ | same |
| G3 dt halving | ≤ 0.6% ✅ | same |
| G5 energy above 4 rad/s (< 2%) | max 1.13% ✅. Energy at 2.2–4 rad/s is still up to 9.9% in soft, high-sea cases: now radiation-damped model physics (connector-mode superharmonic response, springing-like) | `step4fix_out_of_band.log` |
| S1 sensitivity, HF damping ×0.5 / ×2, 40 cases, criterion p90 \|Δr_σ\| < 0.02 fixed in advance | p90 **0.0057** (soft 0.0063, stiff 0.0031); r_max p90 0.016 ✅ | `step4fix_hf_sensitivity.log` |

**Correction to the diagnosis above:** the soft-connector residual changed little once damping was added (e.g. +0.135 → +0.131),
and it is insensitive to a ×4 range of that damping. It is **mainly real model physics, not an artifact**. v1 still
stays quarantined, because it was produced with no damping above 2.2 rad/s, and that could not be shown at the time.

## 2026-11-24: Step 4, dataset v2 (final)

`python src/make_dataset.py --n 1200 --workers 30` with `OMP_NUM_THREADS=1` → `data/dataset.csv`, 21 min
(`logs/step4_dataset_v2_run.log`). Same design, same seeds and same linear values as v1 (asserted in `logs/step4_v1_vs_v2.log`).
Checks (`logs/step4_checks.log`): **ALL PASS**. D1 no NaN · D2 mean σ_Lr/σ_L − 1 = 0.0002 · D3 ≥ 120 per decile, 240 per gap.

**What the data says** (`figures/step4_dataset.png`):
- Potential flow explains **99.68%** of the variance of ln σ_NL. The residual on σ is small: r_σ from −6% to +5% (p05–p95).
  Stiff connectors (k > 10^6.33) sit at −4.1%, mostly drag. Soft connectors (k < 10^5.67) sit at +1.4%, up to +10%.
- The residual on the **30-min peak** is large: r_max from −10% to +45%, up to +55% on soft connectors. **Peaks are where
  potential flow fails here**, which matches the outline's RQ2 (error on peak connector loads).
- The step at log10 k ≈ 5.8 survives the fix, and S1 shows it is insensitive to the HF damping, so it is model physics.
  It is consistent with the connector mode (2.9 rad/s at 10^5.8, `logs/step4_diag_connector_mode.log`) leaving the range
  reached by superharmonics of the wave band. This is an interpretation; it has not been separately demonstrated.
- v2 − v1: r_σ shifts by −0.004 on average (soft −0.0065, stiff −0.0027). The damping fix changed the truth only slightly.

**Implication for step 5:** sampling noise is comparable to the σ residual. One 30-min record gives σ to about 2.8%
(D2), and the 30-min maximum is noisier still. Direct and residual models must therefore be scored against the same noisy
truth, and the noise floor must be reported alongside the errors.

## 2026-11-24: Step 5, run A (pre-registered protocol; protocol in the `src/train_compare.py` header, fixed before training)

`python src/train_compare.py` → `results/step5A_results.csv`, summary `logs/step5_summary.log`. 192 runs, 30 s.
Noise floor (RMSE ln): peak 0.0966, σ 0.0279.

**Peak (main target), RMSE ln, test_in, mean of 5 repeats:**
PF 0.174 at every N · D-raw 0.262 → 0.120 (N=200) → 0.101 (N=600) · D-pf 0.227 → 0.128 → 0.096 · RES 0.219 → 0.169 → 0.104.
Unseen gap 3 m: PF 0.211 · at N=600, D-raw 0.110, D-pf 0.109, RES 0.115.
**σ:** PF 0.047 · at N=600, D-raw 0.033, D-pf 0.030, RES 0.030 (floor 0.028). Unseen gap: RES 0.031, D-pf 0.031, D-raw 0.033.
Conformal at N=600, 90% target: peak coverage in/gap3 RES 0.94/0.89, D-pf 0.92/0.87, D-raw 0.94/0.92 · σ 0.90–0.92 / 0.88–0.90.

**Reading, as it stands:**
1. The outline's hypothesis (RQ2: residual beats direct at equal budget) is **not supported for the peak**. RES is never
   better than D-pf, and at N ≤ 200 it is the worst learned model.
2. Every learned model is **worse than potential flow alone for N < 100**. The fixed recipe (2000 epochs, no early
   stopping, 64-wide) overfits at small N. This is a weakness of the protocol, and it also counts against RES, whose zero-initialised
   start equals PF and is then trained away from it.
3. At N ≥ 400 every learned model reaches the noise floor: about 45% less peak error than potential flow (0.174 → 0.096).
4. On σ, RES ≈ D-pf > D-raw, and all of them sit within 7% of the noise floor at N ≥ 400.
5. Conformal coverage holds in distribution (0.92–0.97). On the unseen gap it drops to 0.87–0.89 for the PF-feature models,
   which is expected: the unseen configuration breaks exchangeability.

## 2026-11-24: Step 5, run B. ⚠️ POST-HOC, designed after seeing run A

Motivation: finding 2 above (overfitting at small N). **One change, applied identically to D-raw, D-pf and RES:** each training
set is split 80/20 (at least 5 validation cases, split seed = repeat). Training runs up to 2000 epochs with validation
loss checked every 10 epochs, and the weights with the lowest validation loss are kept. Everything else is unchanged.
Run A remains the pre-registered result. Run B is reported next to it, never in place of it.

Refactor check: after `--protocol` was added, protocol A re-run reproduces the original run-A summary to every printed
decimal (`logs/step5_refactor_check.log`). (My first check, a `cmp` of a file against itself, proved nothing; it was replaced.)

**Run B results** (`results/step5B_results.csv`, `logs/step5B_summary.log`, 28 s). RMSE ln, mean ± sd over 5 repeats:

| Peak, test_in | N=25 | 50 | 100 | 200 | 400 | 600 |
|---|---|---|---|---|---|---|
| PF | 0.174 | 0.174 | 0.174 | 0.174 | 0.174 | 0.174 |
| D-raw | 0.191 ± 0.023 | 0.167 | 0.129 | 0.122 | 0.105 | 0.102 |
| D-pf | 0.159 ± 0.030 | 0.144 | 0.124 | 0.113 | 0.099 | 0.097 |
| RES | 0.159 ± 0.010 | 0.147 | 0.127 | 0.120 | 0.100 | 0.098 |

| σ, test_in | N=25 | 50 | 100 | 200 | 400 | 600 |
|---|---|---|---|---|---|---|
| PF | 0.047 | 0.047 | 0.047 | 0.047 | 0.047 | 0.047 |
| D-raw | 0.097 ± 0.013 | 0.072 | 0.042 | 0.039 | 0.034 | 0.033 |
| D-pf | 0.053 ± 0.009 | 0.046 | 0.036 | 0.033 | 0.030 | 0.030 |
| RES | 0.042 ± 0.004 | 0.038 | 0.037 | 0.032 | 0.030 | 0.030 |

Unseen gap 3 m, peak at N=25: D-pf 0.147, RES 0.161, D-raw 0.165, PF 0.211. All models converge to about 0.110 at N=600.
Conformal at N=600 (90%): peak coverage in/gap3 RES 0.93/0.88 · D-pf 0.93/0.87 · D-raw 0.95/0.91.
Figure: `figures/step5_learning_curves.png` (top row run A, bottom row run B).

## Step 5 conclusions (what the evidence supports, and what it does not)

1. **Potential flow as prior knowledge cuts the data needed.** With early stopping (run B), both PF-informed models
   beat potential flow alone from N = 25. Raw-direct needs N ≈ 50–100 for the peak. For σ, RES at N = 25 (0.042) equals
   raw-direct at N = 100 (0.042): **about 4× less data**.
2. **Residual vs physics-as-input:** on the peak, RES and D-pf are **indistinguishable** within the 5-repeat spread. RES
   varies 3× less at N = 25 (sd 0.010 vs 0.030). On σ, RES is best at N ≤ 50. **The claim "residual beats direct" holds only
   against raw-input direct learning, not against a direct model that is given the potential-flow features.**
3. **Training protocol matters as much as architecture.** Without early stopping (run A, pre-registered), every learned
   model is worse than potential flow below N = 100, and RES is worst. A residual model is only safe at small N when
   it is regularised to stay near its physics prior.
4. **At N ≥ 400 every learned model reaches the noise floor** (peak 0.097, σ 0.028): roughly half the error of potential flow.
   Past that point a 30-min statistic cannot be predicted better, and no model differences can be resolved.
5. **Unseen configuration (gap 3 m):** all models generalise with errors about 10% higher. Conformal coverage drops from about 0.93
   to 0.87–0.91, as expected when exchangeability is broken; the intervals need recalibration for new configurations.
6. **Limits:** synthetic truth with assumed coefficients; two bodies, head seas, 4 inputs; 5 repeats (small-N
   differences are indicative, not significant); run B is post-hoc.

## 2026-11-24: Step 6, documents for publication

- `README.md`: limits first, results, method with equations, verification table, reproduction, layout, references.
  Voice checks: 0 em dashes, 0 contractions, mean sentence 12 words. Facts checked against the profile (name, degree).
- `report/residual_surrogate_report.pdf`: 10-page technical report with every equation, check, failed run and result,
  typeset with matplotlib (`src/report_engine.py`, `src/make_report.py`; no LaTeX on this machine). Text layer checked
  with pypdf: 17,855 characters, no NUL, every probe found. Every page was inspected as an image; one overflowing
  equation was split.
- `figures/setup_schematic.png`, `figures/headline_result.png` (`src/make_figures.py`).
- `LICENSE` (MIT), `requirements.txt`, `run_all.sh` (full reproduction in the order of this log).
- Privacy scan of all public files: no private data. Repository 2.8 MB without the local wheel cache (ignored).
- Git: repository-local identity, first commit 9523c6e. **Push is done by the author**, after review.
