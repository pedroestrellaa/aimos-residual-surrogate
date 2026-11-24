"""Step 1 - Linear potential-flow coefficients for two barges in tandem (Capytaine BEM).

Geometry: two identical box barges, L x B x T, placed end to end along x with a gap g,
head seas (wave direction beta = 0). Degrees of freedom: heave and pitch of each body (4 DOFs).

Outputs, one NetCDF file per gap: added mass, radiation damping, excitation force
(Froude-Krylov + diffraction), hydrostatic stiffness,
(frequency range capped at 2.2 rad/s, below the first irregular frequency 2.38 rad/s; see make_barge), and the infinite-frequency added mass
needed later by the Cummins time-domain model.

Usage:  python src/bem_two_barges.py --gaps 1 1.5 2 3 4 5 --res 1.0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import xarray as xr
import capytaine as cpt

RHO, G = 1025.0, 9.81
L, B, T = 20.0, 10.0, 2.0          # barge length, beam, draft [m]
KYY = 0.25 * L                      # pitch radius of gyration [m] (assumption, typical barge)


def make_barge(name, x_center, res):
    nx, ny, nz = int(L / res), int(B / res), max(2, int(T / res))
    mesh = cpt.mesh_parallelepiped(size=(L, B, T), center=(x_center, 0.0, -T / 2),
                                   resolution=(nx, ny, nz), missing_sides={"top"}, name=name)
    mass = RHO * L * B * T
    cog = (x_center, 0.0, 0.0)      # CoG at waterline (assumption; only pitch hydrostatics see it)
    # No interior lid. History (logs/): run 1 without lid failed reciprocity/damping-sign checks at
    # omega >= 2.38 rad/s = first irregular frequency. Run 2: generate_lid() returned an empty mesh, silently
    # ignored. Run 3: explicit lid removed the high-frequency failure but changed B33 by 17% at omega = 0.8 rad/s,
    # far from any irregular frequency, and broke mesh convergence (11%). Decision: no lid, omega_max = 2.2 rad/s.
    # Cause found later (LOG, step 4 fix): mesh_rectangle swaps x and y, so the lid was rotated 90 deg. The no-lid
    # result below 2.2 rad/s is confirmed independently by the Haskind relation.
    body = cpt.FloatingBody(mesh=mesh, dofs=cpt.rigid_body_dofs(only=["Heave", "Pitch"], rotation_center=cog),
                            center_of_mass=cog, mass=mass, name=name)
    return body


def solve(gap, res, omegas, out_dir):
    xc = (L + gap) / 2
    b1, b2 = make_barge("b1", -xc, res), make_barge("b2", +xc, res)
    bodies = b1 + b2                 # Multibody: hydrodynamic interaction included
    # hydrostatics and rigid-body inertia per body, then block-diagonal assembly
    K = np.zeros((4, 4)); M = np.zeros((4, 4))
    for i, b in enumerate((b1, b2)):
        Kb = b.compute_hydrostatic_stiffness(rho=RHO, g=G).values
        m = RHO * L * B * T
        K[2*i:2*i+2, 2*i:2*i+2] = Kb
        M[2*i:2*i+2, 2*i:2*i+2] = np.diag([m, m * KYY**2])
    test = xr.Dataset(coords={"omega": np.concatenate([omegas, [np.inf]]),
                              "wave_direction": [0.0],
                              "radiating_dof": list(bodies.dofs),
                              "water_depth": [np.inf], "rho": [RHO], "g": [G]})
    t0 = time.time()
    ds = cpt.BEMSolver().fill_dataset(test, bodies, n_jobs=1)
    elapsed = time.time() - t0
    dofs = [str(d) for d in bodies.dofs]
    sel = dict(radiating_dof=dofs, influenced_dof=dofs)
    A = ds["added_mass"].sel(**sel).values            # (n_omega+1, 4, 4), last row = omega -> inf
    Bd = ds["radiation_damping"].sel(**sel).values
    F = ds["excitation_force"].sel(wave_direction=0.0, influenced_dof=dofs).values  # (n_omega+1, 4) complex, per unit amplitude
    meta = dict(gap=gap, L=L, B=B, T=T, kyy=KYY, rho=RHO, g=G, panel_size=res,
                n_panels=int(bodies.mesh.nb_faces), solve_seconds=round(elapsed, 1), dofs=dofs,
                capytaine=cpt.__version__)
    path = out_dir / f"bem_gap{gap:g}_res{res:g}.npz"
    np.savez(path, omega=omegas, A=A[:-1], B=Bd[:-1], A_inf=A[-1], F=F[:-1], K=K, M=M, meta=json.dumps(meta))
    return meta, path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gaps", type=float, nargs="+", default=[2.0])
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--nw", type=int, default=60)
    a = ap.parse_args()
    out = Path(__file__).resolve().parents[1] / "data"; out.mkdir(exist_ok=True)
    omegas = np.linspace(0.2, 2.2, a.nw)
    for g in a.gaps:
        meta, p = solve(g, a.res, omegas, out)
        print(f"gap={g:g} m  panels={meta['n_panels']}  solve={meta['solve_seconds']} s  -> {p.name}")
