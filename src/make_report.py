"""Technical report (PDF): every step of the mini-project, with equations, checks, failed runs and results.
Usage: python src/make_report.py  ->  report/residual_surrogate_report.pdf"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_engine import Doc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
F = ROOT / "figures"
(ROOT / "report").mkdir(exist_ok=True)
d = Doc(ROOT / "report" / "residual_surrogate_report.pdf",
        "Physics first, learn the remainder: connector loads of two floating modules",
        png_dir=sys.argv[sys.argv.index("--png") + 1] if "--png" in sys.argv else None)

# ------------------------------------------------------------------ title
d.heading("Physics first, learn the remainder", 0)
d.para("A small test of residual learning on potential flow, for connector loads of two floating modules", size=12.5)
d.space(0.05)
d.para("Pedro Henrique de Melo Estrêlla · Technical report · 24 November 2026", size=10)
d.para("Code, data, logs and the step-by-step record: github.com/pedroestrellaa/aimos-residual-surrogate", size=9.5,
       color="0.25")
d.para("Development note: the code was written with an AI coding assistant (Claude, Anthropic) under my direction. "
       "Every step, acceptance criterion, failed run and decision is recorded in LOG.md, with the original output in logs/.",
       size=8.8, color="0.3", style="italic")

d.heading("Summary", 1)
d.para("Two barges are connected by a spring. Linear potential flow, solved with the Capytaine boundary element code, "
       "predicts the connector load in irregular seas. A time-domain model on the same coefficients, with three "
       "nonlinear terms added, plays the role of the truth. Neural networks then learn the connector load at equal "
       "training budget, with and without potential flow as prior knowledge.")
d.bullets([
    "Potential flow as prior knowledge cuts the data needed. For the load standard deviation, a residual model with 25 "
    "cases matches a raw-input model with 100 cases.",
    "On the 30-min peak load, a residual model and a direct model given the potential-flow features tie. The residual "
    "model only varies less between repeats.",
    "In the pre-registered run, without early stopping, every network was worse than potential flow alone below 100 "
    "cases. A residual model is safe at small N only when it is kept close to its physics prior.",
    "With 400 cases or more, every network reaches the noise floor of a 30-min record: about half the peak error of "
    "potential flow.",
])
d.heading("What this is and what it is not", 2)
d.para("The truth is synthetic. Its nonlinear coefficients are assumed, not measured. It is not CFD-FEM and not "
       "experimental data. The study tests a method: how much training data a network needs when potential flow is "
       "given, against when it is not. It says nothing about the true size of real nonlinear loads. The case has two "
       "bodies, head seas and four inputs.")

# ------------------------------------------------------------------ 1 setup
d.heading("1  Model problem", 1)
d.image(F / "setup_schematic.png", caption="Figure 1. Two box barges end to end, a spring connector across the gap, "
        "head seas. Each body moves in heave z and pitch θ.")
d.table([
    ["Quantity", "Value", "Note"],
    ["Body size L × B × T", "20 × 10 × 2 m", "two identical boxes"],
    ["Mass m", "$\\rho L B T$ = 410 t", "freely floating"],
    ["Pitch radius of gyration", "0.25 L = 5 m", "assumed"],
    ["Centre of gravity", "at the waterline", "assumed"],
    ["Gap g", "1.5, 2, 3, 4, 5 m", "gaps ≤ 1 m dropped (panel ≥ gap)"],
    ["Connector stiffness k", "$10^5$ to $10^7$ N/m", "log-uniform"],
    ["Sea states", "$H_s$ 0.5 to 4 m, $T_p$ 6 to 12 s", "JONSWAP, γ = 3.3"],
    ["Water", "deep, $\\rho$ = 1025 kg/m³", ""],
], [2.2, 2.2, 2.6])
d.para("The state vector is $\\mathbf{x} = [z_1, \\theta_1, z_2, \\theta_2]^T$. Pitch is a right-handed rotation about "
       "+y, so a point at distance $x - x_i$ from the centre of body i moves vertically by $z_i - (x - x_i)\\theta_i$. "
       "The facing ends sit at $+L/2$ on body 1 and $-L/2$ on body 2, so the connector stretch is")
d.eq(r"$\delta = z_2 + \frac{L}{2}\theta_2 - z_1 + \frac{L}{2}\theta_1 = \mathbf{c}^T\mathbf{x}, \qquad "
     r"\mathbf{c} = [-1,\ L/2,\ 1,\ L/2]^T$", label="1")

# ------------------------------------------------------------------ 2 BEM
d.heading("2  Step 1: linear potential flow", 1)
d.para("Capytaine 3.0 solves the radiation and diffraction problems for the two bodies together, as one multibody, so "
       "that hydrodynamic interaction is included. It returns the added mass $\\mathbf{A}(\\omega)$, the radiation "
       "damping $\\mathbf{B}(\\omega)$ and the excitation force $\\hat{\\mathbf{F}}_{exc}(\\omega)$ per unit wave "
       "amplitude. The hydrostatic stiffness is $\\mathbf{C}$. With time dependence $e^{-i\\omega t}$, the convention of "
       "Capytaine, the response and the connector force transfer function are")
d.eq(r"$\left[-\omega^2(\mathbf{M}+\mathbf{A}) - i\omega\mathbf{B} + \mathbf{C} + k\,\mathbf{c}\mathbf{c}^T\right]"
     r"\hat{\mathbf{x}} = \hat{\mathbf{F}}_{exc}, \qquad H_F(\omega) = k\,\mathbf{c}^T\hat{\mathbf{x}}(\omega)$", label="2")
d.para("Five checks were fixed before the first run. None was loosened later.")
d.table([
    ["Check", "Definition", "Tolerance"],
    ["C1 reciprocity", "$A_{ij} = A_{ji}$, $B_{ij} = B_{ji}$", "1%"],
    ["C2 identical bodies", "radiation diagonal of body 1 equals that of body 2", "1%"],
    ["C3 long-wave limit", "$|F_{heave}| / C_{33} \\rightarrow 1$ as $\\omega \\rightarrow 0$", "5%"],
    ["C4 mesh convergence", "1.0 m against 0.5 m panels, $A_{33}$, $B_{33}$, $|F_3|$", "5%"],
    ["C5 damping sign", "$B_{ii} \\geq 0$", "exact"],
], [1.6, 4.0, 1.0])
d.heading("Run history", 2)
d.table([
    ["Run", "Change", "Outcome"],
    ["1", "ω 0.2 to 2.5 rad/s, no lid", "C1, C5 fail at ω ≥ 2.38 rad/s, the first irregular frequency"],
    ["2", "interior lid by generate_lid()", "no change: the lid came back empty and was ignored"],
    ["3", "lid built by hand", "C4 fails, B33 off by 17% at 0.8 rad/s (cause found in step 4: x and y swapped)"],
    ["4", "no lid, ω ≤ 2.2 rad/s, gap 0.5 m dropped", "all pass except gap 1 m, C1 = 1.02%"],
    ["4b", "gap 1 m dropped", "all checks pass"],
], [0.5, 2.6, 3.9])
d.para("Final values: reciprocity ≤ 0.49%, mesh convergence ≤ 0.7%, long-wave ratio 0.959. Figure 2 shows that the "
       "interaction between the bodies weakens as the gap opens, that the connector load peaks near heave resonance, "
       "and that above $k \\approx 10^6$ N/m the connector acts as rigid. The sharp feature near 1.9 rad/s for small gaps "
       "is the gap resonance of potential flow, which has no viscous damping.")
d.image(F / "step1_bem_checks.png", caption="Figure 2. Interaction added mass between the two heave modes, and the "
        "connector force transfer function against stiffness and gap.")

# ------------------------------------------------------------------ 3 linear FD
d.heading("3  Step 2: linear prediction in irregular seas", 1)
d.para("The JONSWAP spectrum is")
d.eq(r"$S(\omega) = \alpha\,\omega^{-5}\exp\left[-\frac{5}{4}\left(\frac{\omega_p}{\omega}\right)^4\right]"
     r"\gamma^{\,r}, \quad r = \exp\left[-\frac{(\omega-\omega_p)^2}{2\sigma^2\omega_p^2}\right], \quad "
     r"\sigma = 0.07\ (\omega \leq \omega_p),\ 0.09\ (\omega > \omega_p)$", size=11, label="3")
d.para("with $\\omega_p = 2\\pi/T_p$, $\\gamma = 3.3$, and $\\alpha$ fixed so that $m_0 = H_s^2/16$ on a wide "
       "frequency grid. The spectral moments of the connector force, its standard deviation, its mean zero-upcrossing "
       "period and the Rayleigh expected maximum over a record of duration $T$ are")
d.eq(r"$m_{n,F} = \int \omega^n |H_F(\omega)|^2 S(\omega)\,d\omega, \quad \sigma_F = \sqrt{m_{0,F}}, \quad "
     r"T_z = 2\pi\sqrt{m_{0,F}/m_{2,F}}$", label="4")
d.eq(r"$\mathbb{E}[F_{max}] \approx \sigma_F\left(\sqrt{2\ln N} + \frac{0.5772}{\sqrt{2\ln N}}\right), \qquad "
     r"N = T/T_z,\ T = 1800\ \mathrm{s}$", label="5")
d.table([
    ["Check", "Result"],
    ["L1 normalisation $4\\sqrt{m_0} = H_s$", "error ≤ 2e-16"],
    ["L2 wave energy outside 0.2 to 2.2 rad/s", "4.0% at $T_p$ = 6 s, 0.3% at 12 s (all below 5%)"],
    ["L3 frequency grid 60 against 600 points", "max 0.10% over 27 cases"],
    ["L4 independent Monte-Carlo, 20 records of 3 h", "σ within 0.14 to 0.26%"],
    ["L5 zero-stiffness limit", "ratio 1.5e-8"],
], [3.4, 3.6])
d.para("L3 first passed silently: it printed only on failure, so silence was the evidence. It was rewritten to print "
       "its maximum error. The linear model is exactly proportional to $H_s$, so any other dependence on wave height "
       "must come from the nonlinear truth.")
d.image(F / "step2_linear_fd.png", caption="Figure 3. Linear standard deviation of the connector force over peak period "
        "and stiffness for two gaps, and the linear expected 30-min maximum.")

# ------------------------------------------------------------------ 4 Cummins
d.heading("4  Step 3: synthetic nonlinear truth", 1)
d.para("The time-domain model is the Cummins equation on the same coefficients, with three nonlinear terms:")
d.eq(r"$(\mathbf{M}+\mathbf{A}_\infty)\ddot{\mathbf{x}} + \int_0^t\mathbf{K}(t-\tau)\dot{\mathbf{x}}(\tau)\,d\tau + "
     r"\mathbf{C}\mathbf{x} = \mathbf{F}_{exc}(t) - \mathbf{c}\,F_c(\delta) + \mathbf{F}_{drag} - "
     r"\mathbf{c}\,c_g|\dot\delta|\dot\delta$", size=11.5, label="6")
d.eq(r"$\mathbf{K}(t) = \frac{2}{\pi}\int_0^{\omega_{max}}\mathbf{B}(\omega)\cos(\omega t)\,d\omega, \qquad "
     r"\mathbf{A}_\infty = \mathbf{A}(\omega) + \frac{1}{\omega}\int_0^\infty\mathbf{K}(t)\sin(\omega t)\,dt$", label="7")
d.para("The second relation, by Ogilvie, is evaluated at every frequency from 0.4 to 2.0 rad/s and averaged. Its spread "
       "across frequency checks the kernel. The wave and the excitation are built from random phases $\\varphi_j$ on "
       "2000 jittered frequencies:")
d.eq(r"$a_j = \sqrt{2S(\omega_j)\Delta\omega}, \qquad \mathbf{F}_{exc}(t) = \mathrm{Re}\sum_j a_j e^{i\varphi_j}"
     r"\hat{\mathbf{F}}_{exc}(\omega_j)\,e^{-i\omega_j t}$", label="8")
d.para("The same phases give the linear frequency-domain force record, so the linear and the nonlinear responses are "
       "compared on the identical sea. Integration is fourth-order Runge-Kutta at $\\Delta t$ = 0.05 s, with the memory "
       "integral held constant within a step. Each case runs 1800 s after a 200 s discard and a 50 s ramp.")
d.heading("Nonlinear terms (assumed coefficients)", 2)
d.eq(r"$F_c(\delta) = k\,\delta\left[1 + (\delta/\delta_0)^2\right], \qquad \delta_0 = 0.5\ \mathrm{m}$", label="9")
d.eq(r"$w(x) = \dot z_i - (x - x_i)\dot\theta_i, \qquad f(x) = -\frac{1}{2}\rho C_D B\,|w|w, \qquad C_D = 1$",
     size=11.5, label="10a")
d.eq(r"$F_{z,i} = \int f(x)\,dx, \qquad M_{\theta,i} = -\int (x - x_i)\, f(x)\,dx$", size=11.5, label="10b")
d.eq(r"$c_g = 5\times 10^4\ \frac{1.5\ \mathrm{m}}{g}\ \mathrm{kg/m}$", label="11")
d.para("The drag acts strip by strip on the flat bottom (20 strips per body). The gap term damps the relative "
       "vertical velocity of the facing ends, more strongly at small gaps. It mimics the viscous damping of the gap "
       "resonance that potential flow overpredicts.")
d.heading("Gates", 2)
d.table([
    ["Gate", "Criterion", "Result"],
    ["G1 kernel", "Ogilvie spread < 3%, $\\mathbf{A}_\\infty$ vs BEM < 3%, tail < 2%", "≤ 1.2%, ≤ 0.7%, ≤ 0.7%"],
    ["G2 linear gate, 12 cases", "nonlinear terms off: σ within 3%, record RMS within 5%", "σ ≤ 1.5%, RMS ≤ 2.2%"],
    ["G3 time step", "Δt 0.05 against 0.025 s, all terms on", "≤ 0.6%"],
], [1.7, 3.4, 1.9])
d.para("The first linear comparison failed: σ was 17% low and the correlation was 0.80. The records had been built "
       "with $e^{+i\\omega t}$, which pairs each excitation with the complex conjugate of its response. After the fix "
       "the correlation was 0.99995. The first choice of $\\delta_0$ = 0.2 m multiplied σ by ten on soft connectors. "
       "That truth would replace potential flow, not correct it. A criterion was then fixed before a scan: "
       "$\\sigma_{NL}/\\sigma_L$ within [0.6, 1.8] at every corner of the design space, taking the smallest "
       "$\\delta_0$ that meets it. The result was 0.5 m.")
d.image(F / "step3_time_domain.png", caption="Figure 4. Top: the linear gate, where the two curves coincide. Bottom: "
        "the synthetic truth, with sharper peaks from the hardening connector.")

# ------------------------------------------------------------------ 5 dataset + fix
d.heading("5  Step 4: dataset, a defect, and the fix", 1)
d.para("The design is a Latin hypercube of 1200 cases over $H_s$, $T_p$ and $\\log_{10}k$, with the gap cycling "
       "through its five values and one seed per case. The first dataset passed every data check, but showed a sharp "
       "step in the residual at $k \\approx 10^{5.8}$ N/m. The natural frequencies of the coupled system")
d.eq(r"$\left[\mathbf{C} + k\,\mathbf{c}\mathbf{c}^T - \omega_n^2\left(\mathbf{M} + \mathbf{A}(\omega_n)\right)"
     r"\right]\boldsymbol{\phi} = 0$", label="12")
d.para("place the connector mode at 1.84 rad/s for $k = 10^5$ and at 2.95 rad/s for $k = 10^{5.8}$, above the band "
       "where the kernel had damping. A criterion was fixed: less than 2% of the force variance outside the band "
       "with damping. Soft connectors in high seas put 4 to 11% there. The first dataset was quarantined.")
d.heading("Which damping is right: the Haskind relation", 2)
d.para("Steps 1 and 3 left an open question: without and with a lid, the heave damping differed. The Haskind "
       "relation gives the damping from the excitation force over all headings, which is energy conservation and "
       "independent of the radiation solution. In deep water,")
d.eq(r"$B_{33}(\omega) = \frac{k}{8\pi\rho g\,c_g}\int_0^{2\pi}|X_3(\omega,\beta)|^2\,d\beta, \qquad "
     r"k = \frac{\omega^2}{g}, \quad c_g = \frac{g}{2\omega}$", label="13")
d.table([
    ["ω [rad/s]", "0.8", "1.2", "1.6", "2.0", "2.2", "2.6", "2.8", "3.2"],
    ["no lid, 1.0 m panels", "1.035", "1.037", "1.050", "1.111", "", "", "1.074", ""],
    ["lid as first built (axes swapped)", "0.884", "1.300", "1.386", "0.520", "", "", "0.784", ""],
    ["corrected lid, 0.5 m panels", "", "", "1.057", "", "1.055", "1.258", "1.285", "1.209"],
], [2.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
d.para("Ratio of radiation damping to the Haskind value, single body. The cause of every lid failure was found in "
       "the source of Capytaine itself: mesh_rectangle(size=(a, b)) places a along y. The lid was rotated by 90° "
       "and reached 5 m beyond each side of the hull. With the axes swapped back, the lid changes the damping by at "
       "most 0.7% below the first irregular frequency, as it must.")
d.heading("Fix: radiation damping up to 4 rad/s", 2)
d.para("Below 2.2 rad/s the kernel uses the validated two-body coefficients. Above, it uses single-body coefficients "
       "from a 0.5 m mesh with the corrected lid. Interaction between the bodies is neglected there, because those "
       "waves are shorter than 13 m. With a linear blend weight $a(\\omega)$ over 2.0 to 2.2 rad/s and a cosine taper "
       "$\\tau(\\omega)$ over 3.8 to 4.0 rad/s,")
d.eq(r"$\mathbf{B}_{kernel}(\omega) = \tau(\omega)\left[(1-a)\,\mathbf{B}_{2\,body}(\omega) + a\,s_{hf}\,"
     r"\mathrm{blockdiag}\left(\mathbf{B}_{1\,body}(\omega)\right)\right]$", label="14")
d.para("The factor $s_{hf}$ is 1. It was set to 0.5 and to 2 to test sensitivity. The criterion, fixed in advance, "
       "was a 90th percentile change of $r_\\sigma = \\ln(\\sigma_{NL}/\\sigma_{L})$ below 0.02 over 40 cases. "
       "The result was 0.0057. All gates were rerun and pass, and energy above 4 rad/s is at most 1.13%. The residual "
       "moved by 0.004 on average, so the soft-connector effect is mainly model physics. The first dataset stays "
       "quarantined, because that could not be shown at the time.")
d.heading("Final dataset", 2)
d.table([
    ["Quantity", "Value"],
    ["Cases", "1200, 240 per gap, at least 120 per decile of each input"],
    ["Variance of $\\ln\\sigma_{NL}$ explained by $\\ln\\sigma_L$", "99.68%"],
    ["$r_\\sigma = \\ln(\\sigma_{NL}/\\sigma_L)$, 5th to 95th percentile", "-6% to +5%"],
    ["$r_{max} = \\ln(F_{max,NL}/F_{max,L})$, 5th to 95th percentile", "-10% to +45% (up to +55% on soft connectors)"],
    ["Sampling noise of σ from one 30-min record", "2.8%"],
], [3.7, 3.3])
d.image(F / "step4_dataset.png", caption="Figure 5. Truth against potential flow, the residual against stiffness, and "
        "the residual against wave height for each gap.")

# ------------------------------------------------------------------ 6 learning
d.heading("6  Step 5: direct against residual learning", 1)
d.para("Splits: all 240 cases with a 3 m gap form an unseen test set. Of the rest, 192 cases are an in-distribution "
       "test set, 168 a calibration set, and 600 the training pool. Training sets are nested random subsets of N = 25 "
       "to 600 cases, repeated five times. The targets are $y = \\ln F_{max}$ (main) and $y = \\ln\\sigma$.")
d.table([
    ["Model", "Prediction"],
    ["PF, potential flow only", "$\\hat y = \\ln y_{PF}$"],
    ["D-raw, direct", "$\\hat y = f_\\theta(H_s, T_p, g, \\log_{10}k)$"],
    ["D-pf, direct with features", "$\\hat y = f_\\theta(H_s, T_p, g, \\log_{10}k, \\ln\\sigma_F, \\ln T_z, "
     "\\ln\\mathbb{E}[F_{max}])$"],
    ["RES, residual", "$\\hat y = \\ln y_{PF} + g_\\theta(\\mathrm{same\\ inputs})$, $g_\\theta = 0$ at start"],
], [2.2, 4.8])
d.para("Every network is the same: inputs standardised on the training set, two hidden layers of 64, SiLU, Adam with "
       "learning rate $10^{-3}$ and weight decay $10^{-4}$, full batch, 2000 epochs. The error measure is the RMSE of "
       "$\\ln y$, close to a relative error. A 30-min statistic is random even when the physics is exact. The noise "
       "floor is estimated from the linear model, where the exact expectation is known:")
d.eq(r"$\epsilon_{floor} = \mathrm{RMSE}\left[\ln F_{max,L}^{record} - \ln\mathbb{E}[F_{max,L}]\right] = 0.097"
     r"\ \ (\mathrm{peak}), \quad 0.028\ \ (\sigma)$", size=11, label="15")
d.para("Split-conformal intervals use the absolute residuals $R_i$ on the n calibration cases:")
d.eq(r"$\hat q = R_{(\lceil (n+1)(1-\alpha)\rceil)}, \qquad \hat y \pm \hat q, \qquad \alpha = 0.1$", label="16")
d.heading("Run A, pre-registered", 2)
d.table([
    ["Peak, in-distribution", "N=25", "50", "100", "200", "400", "600"],
    ["PF", "0.174", "0.174", "0.174", "0.174", "0.174", "0.174"],
    ["D-raw", "0.262", "0.192", "0.155", "0.120", "0.105", "0.101"],
    ["D-pf", "0.227", "0.232", "0.175", "0.128", "0.099", "0.096"],
    ["RES", "0.219", "0.230", "0.201", "0.169", "0.115", "0.104"],
], [2.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])
d.para("Every network is worse than potential flow alone below N = 100, and the residual model is the worst learner "
       "up to N = 200. The recipe overfits at small N. This also counts against the residual model, which starts at "
       "potential flow and is then trained away from it.")
d.heading("Run B, post-hoc: early stopping", 2)
d.para("One change, applied to all three learned models: each training set is split 80/20, the validation loss is "
       "checked every 10 epochs, and the best weights are kept. It was declared in the log before it ran.")
d.table([
    ["Peak / σ, in-distribution", "N=25", "50", "100", "200", "400", "600"],
    ["PF", ".174/.047", ".174/.047", ".174/.047", ".174/.047", ".174/.047", ".174/.047"],
    ["D-raw", ".191/.097", ".167/.072", ".129/.042", ".122/.039", ".105/.034", ".102/.033"],
    ["D-pf", ".159/.053", ".144/.046", ".124/.036", ".113/.033", ".099/.030", ".097/.030"],
    ["RES", ".159/.042", ".147/.038", ".127/.037", ".120/.032", ".100/.030", ".098/.030"],
], [2.0, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85])
d.image(F / "step5_learning_curves.png", caption="Figure 6. Learning curves, mean ± sd over 5 repeats. Top row run A, "
        "bottom row run B. Columns: peak and σ, in-distribution and unseen gap.")
d.table([
    ["Conformal, N = 600, 90% target", "Peak coverage in / unseen", "σ coverage in / unseen"],
    ["D-raw", "0.95 / 0.91", "0.93 / 0.90"],
    ["D-pf", "0.93 / 0.87", "0.90 / 0.89"],
    ["RES", "0.93 / 0.88", "0.92 / 0.88"],
], [2.6, 2.2, 2.2])

# ------------------------------------------------------------------ 7 conclusions
d.heading("7  Conclusions", 1)
d.bullets([
    "Potential flow as prior knowledge, as a backbone or as an input, cut the data needed. For σ the residual model "
    "with 25 cases equals the raw-input model with 100 cases.",
    "On the peak load the residual model and the direct model with potential-flow features are indistinguishable within "
    "five repeats. The residual model varies three times less at N = 25.",
    "A residual model is safe at small N only when it is kept near its physics prior. Without early stopping it was the "
    "worst learner.",
    "Beyond about 400 cases every network reaches the noise floor, about half the peak error of potential flow.",
    "On the unseen 3 m gap every model generalises with 5 to 12% more error at N = 600. Conformal coverage drops to "
    "0.87 to 0.91 and needs recalibration for a new configuration.",
])
d.heading("Limits", 2)
d.bullets([
    "The truth is synthetic, with assumed coefficients ($C_D$, $\\delta_0$, $c_g$).",
    "Two rigid bodies, head seas, four inputs. No flexibility, no mooring, no oblique seas.",
    "Damping above 2.2 rad/s neglects interaction between the bodies. Its effect is bounded by the sensitivity test.",
    "Five repeats: differences at small N are indicative, not statistically significant. Run B is post-hoc.",
])
d.heading("Natural next steps", 2)
d.bullets([
    "Replace the synthetic truth with coupled CFD-FEM or basin data for a subset of cases chosen by potential flow.",
    "N modules, with the exact constraints built in: equal and opposite connector forces, and symmetry of identical modules.",
    "A decision rule that returns a case to the full solver when the input leaves the training envelope.",
])

# ------------------------------------------------------------------ 8 provenance
d.heading("8  Reproduction and provenance", 1)
d.para("bash run_all.sh reproduces every number and figure in about 30 minutes on 30 cores. Package versions are "
       "pinned in requirements.lock. All 30 wheels used were checked against the SHA256 hashes published by PyPI and "
       "by the PyTorch index (provenance_wheels.tsv). LOG.md records each step, and logs/ keeps the original output of "
       "every run, including the failed ones listed below.")
d.table([
    ["Failed or superseded run", "What it showed", "Log"],
    ["BEM runs 1 to 3", "irregular frequency; empty lid; rotated lid", "step1_*_RUN1/2/3_*"],
    ["BEM run 4", "gap 1 m at 1.02% reciprocity", "step1_checks_RUN4_*"],
    ["Linear check run 1", "L3 silent on success", "step2_checks_RUN1_*"],
    ["Time-domain smoke 1", "time convention sign", "step3_smoke_RUN1_*"],
    ["Truth with $\\delta_0$ = 0.2 m", "nonlinearity ten times too strong", "step3_checks_RUN1_*"],
    ["Dataset v1", "no damping above 2.2 rad/s", "step4_checks_v1.log"],
    ["Check run stopped after 60 min", "410 MB array per worker, rewritten", "step4fix_td_checks.log"],
], [2.3, 2.9, 1.8])

d.heading("References", 1)
for ref in [
    "Ancellin, M. and Dias, F. (2019). Capytaine: a Python-based linear potential flow solver. Journal of Open Source "
    "Software, 4(36), 1341.",
    "Angelopoulos, A. N. and Bates, S. (2023). Conformal prediction: a gentle introduction. Foundations and Trends in "
    "Machine Learning, 16(4), 494-591.",
    "Cummins, W. E. (1962). The impulse response function and ship motions. Schiffstechnik, 9, 101-109.",
    "Hasselmann, K. et al. (1973). Measurements of wind-wave growth and swell decay during the Joint North Sea Wave "
    "Project (JONSWAP). Deutsche Hydrographische Zeitschrift, A8(12).",
    "Kennedy, M. C. and O'Hagan, A. (2001). Bayesian calibration of computer models. Journal of the Royal Statistical "
    "Society B, 63(3), 425-464.",
    "Newman, J. N. (1977). Marine Hydrodynamics. MIT Press.",
    "Ogilvie, T. F. (1964). Recent progress toward the understanding and prediction of ship motions. 5th Symposium "
    "on Naval Hydrodynamics, Bergen.",
]:
    d.para(ref, size=9, indent=0.0)
d.close()
print("report/residual_surrogate_report.pdf, pages:", d.page)
