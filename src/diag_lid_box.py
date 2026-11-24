"""Diagnostic: hand-built waterplane lid for the box. Criterion fixed before running: B33 lid/no-lid within 2% at
omega 0.8, 1.2, 1.6 rad/s (below the first irregular frequency, where a correct lid must change nothing)."""
import numpy as np, capytaine as cpt
cpt.set_logging("ERROR")
S = cpt.BEMSolver(); L, B, T = 20., 10., 2.
hull = cpt.mesh_parallelepiped(size=(L, B, T), center=(0, 0, -T/2), resolution=(20, 10, 2), missing_sides={"top"})
def b33(lid, w):
    b = cpt.FloatingBody(mesh=hull, lid_mesh=lid, dofs=cpt.rigid_body_dofs(only=["Heave"], rotation_center=(0, 0, 0)), center_of_mass=(0, 0, 0))
    return S.solve(cpt.RadiationProblem(body=b, omega=w, radiating_dof="Heave")).radiation_dampings["Heave"]
ref = {w: b33(None, w) for w in [0.8, 1.2, 1.6]}
for z in [0.0, -0.02, -0.1]:
    for shrink, res in [(1.0, 1.0), (0.98, 1.0), (0.95, 1.0), (1.0, 0.5)]:
        lid = cpt.mesh_rectangle(size=(L*shrink, B*shrink), center=(0, 0, z), resolution=(int(L/res), int(B/res)), normal=(0, 0, 1))
        r = [b33(lid, w) / ref[w] for w in ref]
        ok = max(abs(x - 1) for x in r) < 0.02
        print(f"lid z={z:<5} size x{shrink:<4} panel {res} m:  B33 lid/nolid " + ", ".join(f"{x:.3f}" for x in r) + f"   {'PASS' if ok else 'fail'}", flush=True)
