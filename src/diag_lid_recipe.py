"""Diagnostic: does the lid work when built by Capytaine's own recipe (generate_lid on the immersed part)?
Test on a vertical cylinder and on the box; compare B33 lid vs no lid at omega where no irregular frequency exists."""
import numpy as np, capytaine as cpt
cpt.set_logging("ERROR")
S = cpt.BEMSolver()
def b33(mesh, lid, w):
    b = cpt.FloatingBody(mesh=mesh, lid_mesh=lid, dofs=cpt.rigid_body_dofs(only=["Heave"], rotation_center=(0, 0, 0)), center_of_mass=(0, 0, 0))
    return S.solve(cpt.RadiationProblem(body=b, omega=w, radiating_dof="Heave")).radiation_dampings["Heave"]
cyl = cpt.mesh_vertical_cylinder(length=4.0, radius=5.0, center=(0, 0, 0), resolution=(8, 30, 10)).immersed_part()
box_closed = cpt.mesh_parallelepiped(size=(20, 10, 4), center=(0, 0, 0), resolution=(20, 10, 4)).immersed_part()
box_open = cpt.mesh_parallelepiped(size=(20, 10, 2), center=(0, 0, -1), resolution=(20, 10, 2), missing_sides={"top"})
for name, m in [("cylinder (closed, clipped)", cyl), ("box closed, clipped", box_closed), ("box open-top (mine)", box_open)]:
    for z in [0.0, -0.05]:
        try:
            lid = m.generate_lid(z=z); n = lid.nb_faces
        except Exception as e:
            lid, n = None, f"ERR {e}"
        line = f"{name:<27} generate_lid(z={z}): faces {n}"
        if lid is not None and lid.nb_faces > 0:
            r = [b33(m, lid, w) / b33(m, None, w) for w in [0.8, 1.2, 1.6]]
            line += "   B33 lid/nolid at 0.8,1.2,1.6: " + ", ".join(f"{x:.3f}" for x in r)
        print(line, flush=True)
    print(f"{'':<27} nb_faces hull {m.nb_faces}, z range {m.vertices[:,2].min():.2f}..{m.vertices[:,2].max():.2f}, normals z of bottom faces mean {m.faces_normals[m.faces_centers[:,2]<-1.9,2].mean():+.2f}")
