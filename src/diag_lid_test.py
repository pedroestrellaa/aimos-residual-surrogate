import numpy as np, capytaine as cpt, logging, sys
cpt.set_logging("ERROR")
L,B,T,RHO=20.,10.,2.,1025.
def body(res, lidz):
    nx,ny,nz=int(L/res),int(B/res),max(2,int(T/res))
    m=cpt.mesh_parallelepiped(size=(L,B,T),center=(0,0,-T/2),resolution=(nx,ny,nz),missing_sides={"top"})
    lid=None if lidz is None else cpt.mesh_rectangle(size=(L,B),center=(0,0,lidz),resolution=(nx,ny))
    return cpt.FloatingBody(mesh=m,lid_mesh=lid,dofs=cpt.rigid_body_dofs(only=["Heave"],rotation_center=(0,0,0)),center_of_mass=(0,0,0))
ws=np.array([0.8,1.2,1.6,2.0])
S=cpt.BEMSolver()
for lidz in [None,-0.02,-0.1,-0.25]:
    row=[]
    for res in [1.0,0.5]:
        b=body(res,lidz)
        row.append(np.array([S.solve(cpt.RadiationProblem(body=b,omega=w,radiating_dof="Heave")).radiation_dampings["Heave"] for w in ws]))
    rel=np.abs(row[0]-row[1])/np.abs(row[1]).max()
    print(f"lid z={lidz}: B33(res1)={np.round(row[0]/1e3)}  B33(res0.5)={np.round(row[1]/1e3)} kNs/m  max rel diff {rel.max():.3f}")
