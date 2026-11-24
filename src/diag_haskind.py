"""Diagnostic: which heave damping is right, no-lid or lid? Independent check by the Haskind relation (deep water)
    B_33 = k / (8 pi rho g c_g) * int_0^{2pi} |X_3(theta)|^2 dtheta ,  k = w^2/g,  c_g = g / (2 w)
X_3 = heave excitation force per unit amplitude (Froude-Krylov + diffraction), 72 headings. Single box 20x10x2 m."""
import numpy as np, capytaine as cpt
cpt.set_logging("ERROR")
L, B, T, RHO, G = 20., 10., 2., 1025., 9.81
def body(res, lidz):
    nx, ny, nz = int(L/res), int(B/res), max(2, int(T/res))
    m = cpt.mesh_parallelepiped(size=(L, B, T), center=(0, 0, -T/2), resolution=(nx, ny, nz), missing_sides={"top"})
    # mesh_rectangle swaps x and y (Capytaine source: "TODO Fix: Exchange x and y"): pass (B, L) to get L along x
    lid = None if lidz is None else cpt.mesh_rectangle(size=(B, L), center=(0, 0, lidz), resolution=(ny, nx))
    return cpt.FloatingBody(mesh=m, lid_mesh=lid, dofs=cpt.rigid_body_dofs(only=["Heave"], rotation_center=(0, 0, 0)),
                            center_of_mass=(0, 0, 0))
S = cpt.BEMSolver(); th = np.linspace(0, 2*np.pi, 72, endpoint=False)
for lidz in [-0.02]:   # run 1 (lid -0.1, axes swapped) kept in logs/step4_diag_haskind.log
    b = body(1.0, lidz)
    for w in [0.8, 1.6, 2.2, 2.385, 2.6, 2.8, 3.2, 3.6, 4.0]:
        Brad = S.solve(cpt.RadiationProblem(body=b, omega=w, radiating_dof="Heave")).radiation_dampings["Heave"]
        X = np.array([S.solve(cpt.DiffractionProblem(body=b, omega=w, wave_direction=t)).forces["Heave"] for t in th])
        X = X + np.array([cpt.bem.airy_waves.froude_krylov_force(cpt.DiffractionProblem(body=b, omega=w, wave_direction=t))["Heave"] for t in th])
        k = w**2/G; cg = G/(2*w)
        Bh = k/(8*np.pi*RHO*G*cg) * np.sum(np.abs(X)**2) * (2*np.pi/len(th))
        print(f"lid {str(lidz):<5} omega {w:.1f}:  B33 radiation {Brad/1e3:8.1f}  Haskind {Bh/1e3:8.1f} kNs/m   ratio {Brad/Bh:.3f}", flush=True)
