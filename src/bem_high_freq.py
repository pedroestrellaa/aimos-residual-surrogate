"""Step 4-fix - Radiation coefficients ABOVE the wave band (2.2-4.0 rad/s), single barge, fine mesh with lid.

Why: the connector mode of soft connectors sits at 2.3-3.5 rad/s. Nonlinear terms feed energy there, and the time-domain
kernel needs real radiation damping at those frequencies (dataset v1 had none -> artifact).
How: one barge (interaction between the bodies is neglected here: waves at > 2.2 rad/s are < 13 m long, and their
interaction coefficients are small), panel 0.5 m, interior lid at z = -0.05 m to suppress the dense irregular
frequencies above 2.38 rad/s. Validated by logs/step4fix_haskind_fine.log (radiation vs Haskind 1.06-1.29 up to 3.2 rad/s).
⛔ Capytaine's mesh_rectangle swaps x and y: size=(B, L) gives L along x. This bug caused every lid failure in steps 1-4.
"""
import json
from pathlib import Path

import numpy as np
import capytaine as cpt

cpt.set_logging("ERROR")
RHO, G = 1025.0, 9.81
L, B, T, RES, LIDZ = 20.0, 10.0, 2.0, 0.5, -0.05

if __name__ == "__main__":
    nx, ny, nz = int(L / RES), int(B / RES), int(T / RES)
    hull = cpt.mesh_parallelepiped(size=(L, B, T), center=(0, 0, -T / 2), resolution=(nx, ny, nz), missing_sides={"top"})
    lid = cpt.mesh_rectangle(size=(B, L), center=(0, 0, LIDZ), resolution=(ny, nx))
    v = lid.vertices
    assert abs(v[:, 0].max() - L / 2) < 1e-9 and abs(v[:, 1].max() - B / 2) < 1e-9, "lid axes wrong"
    body = cpt.FloatingBody(mesh=hull, lid_mesh=lid, center_of_mass=(0, 0, 0),
                            dofs=cpt.rigid_body_dofs(only=["Heave", "Pitch"], rotation_center=(0, 0, 0)))
    omega = np.round(np.arange(2.0, 4.01, 0.1), 3)
    S = cpt.BEMSolver()
    A = np.zeros((omega.size, 2, 2)); Bm = np.zeros((omega.size, 2, 2))
    for n, w in enumerate(omega):
        for j, dof in enumerate(["Heave", "Pitch"]):
            r = S.solve(cpt.RadiationProblem(body=body, omega=w, radiating_dof=dof))
            for i, idof in enumerate(["Heave", "Pitch"]):
                A[n, i, j] = r.added_masses[idof]; Bm[n, i, j] = r.radiation_dampings[idof]
        print(f"omega {w:.1f}: B33 {Bm[n,0,0]/1e3:8.2f} kNs/m  B55 {Bm[n,1,1]/1e6:8.3f} MNms  "
              f"asym {abs(Bm[n,0,1]-Bm[n,1,0])/max(abs(Bm[n]).max(),1):.3f}", flush=True)
    ok = (np.diagonal(Bm, axis1=1, axis2=2) > 0).all()
    print("H1 B_ii > 0 at all frequencies:", "PASS" if ok else "FAIL")
    out = Path(__file__).resolve().parents[1] / "data" / "bem_single_hf.npz"
    np.savez(out, omega=omega, A=A, B=Bm, meta=json.dumps(dict(L=L, B=B, T=T, panel=RES, lid_z=LIDZ,
                                                             capytaine=cpt.__version__)))
    print("wrote", out.name)
