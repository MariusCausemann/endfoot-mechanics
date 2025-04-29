from fenics import *
#from multiphenics import *
import numpy as np
import pyvista as pv
import yaml

parameters["refinement_algorithm"] = "plaza_with_parent_facets"

porous_id = 1
fluid_id = 2


def generate_subdomain_restriction(mesh, subdomains, subdomain_id):
    D = mesh.topology().dim()
    # Initialize empty restriction
    restriction = MeshRestriction(mesh, None)
    for d in range(D + 1):
        mesh_function_d = MeshFunction("bool", mesh, d)
        mesh_function_d.set_all(False)
        restriction.append(mesh_function_d)
    # Mark restriction mesh functions based on subdomain id
    for c in cells(mesh):
        if subdomains[c] == subdomain_id:
            restriction[D][c] = True
            for d in range(D):
                for e in entities(c, d):
                    restriction[d][e] = True
    # Return
    return restriction

def mark_internal_interface(mesh, subdomains, boundaries, side_a, side_b, interface_id):
    # set internal interface
    for f in facets(mesh):
        domains = []
        for c in cells(f):
            domains.append(subdomains[c])
        domains = list(set(domains))
        domains.sort()
        if len(domains) < 2:
            continue
        if domains[0] in side_a and domains[1] in side_b:
            boundaries[f] = interface_id


def set_external_boundaries(mesh, subdomains, boundaries, subdomain_ids,
                            boundary_id, previous_boundary_ids=[]):
    
    for f in facets(mesh):
        domains = []
        for c in cells(f):
            domains.append(subdomains[c])
        domains = list(set(domains))
        if f.exterior() and len(domains) == 1:
            if domains[0] in subdomain_ids and boundaries[f] in previous_boundary_ids:
                boundaries[f] = boundary_id


def mark_pvs(mesh, pvs, pvs_marker=0, pvs_max_dist=250, fluid_id=1):
    mesh.compute_implicit_distance(pvs, inplace=True)
    mesh = mesh.point_data_to_cell_data()
    mesh["gmsh:physical"] = mesh["gmsh:physical"].astype(np.int32)
    pvs_marker = (mesh["implicit_distance"] < pvs_max_dist) & (mesh["gmsh:physical"] ==1)
    subdomains = mesh.cell_data["gmsh:physical"]
    subdomains[pvs_marker] = 0 
    #mesh.cell_data.clear()
    #mesh.point_data.clear()
    mesh.cell_data["subdomains"] = subdomains
    return mesh


def mark_boundaries(mesh, sm, cell_ids, eps=0.02, axis=1):

    bm = MeshFunction("size_t", mesh, 2, 0)

    coords = mesh.coordinates()

    y_min, y_max =  coords[:,axis].min(), coords[:,axis].max()

    eps = (y_max - y_min)*eps

    top = CompiledSubDomain("on_boundary && near(x[axis], y, tol)",
                            y=y_max, tol=eps, axis=axis)
    bottom = CompiledSubDomain("on_boundary && near(x[axis], y, tol)",
                            y=y_min, tol=eps,axis=axis)

    outer = CompiledSubDomain("on_boundary")

    pvs_id = 0
    ecs_id = 1

    top_id = 1
    outer_id = 3
    vessel_id = 4
    bottom_id = 2

    ecs_axial_id = 6
    cells_axial_id = 7


    outer.mark(bm, outer_id)
    top.mark(bm, top_id)
    bottom.mark(bm, bottom_id)

    ecs_axial_id = 6
    cells_axial_id = 7

    ecs_outer_id = 8
    cells_outer_id = 9

    pvs_top_id = 10
    pvs_bottom_id = 11

    membrane_id = 12
    apq_membrane_id = 13

    set_external_boundaries(mesh, sm, bm, [pvs_id], vessel_id,
                            previous_boundary_ids=[outer_id])

    set_external_boundaries(mesh, sm, bm, [ecs_id], ecs_axial_id,
                            previous_boundary_ids=[top_id, bottom_id])

    set_external_boundaries(mesh, sm, bm, [pvs_id], pvs_top_id,
                            previous_boundary_ids=[top_id])

    set_external_boundaries(mesh, sm, bm, [pvs_id], pvs_bottom_id,
                            previous_boundary_ids=[bottom_id])

    set_external_boundaries(mesh, sm, bm, cell_ids,
                            cells_axial_id,
                            previous_boundary_ids=[top_id, bottom_id ])

    set_external_boundaries(mesh, sm, bm, [ecs_id], ecs_outer_id,
                            previous_boundary_ids=[outer_id])

    set_external_boundaries(mesh, sm, bm, cell_ids,
                            cells_outer_id,
                            previous_boundary_ids=[outer_id])

    mark_internal_interface(mesh, sm, bm, side_a=[ecs_id], side_b=cell_ids, 
                            interface_id=membrane_id)

    mark_internal_interface(mesh, sm, bm, side_a=[pvs_id], side_b=cell_ids,
                            interface_id=apq_membrane_id)

    bm.rename("boundaries","")

    return bm


def generate_subdomains(mesh, sm, cell_ids, fluid_ids, mesh_name):
    cell_mask = np.isin(sm.array()[:], cell_ids )
    fluid_mask = np.isin(sm.array()[:],  fluid_ids )

    sm.array()[cell_mask] = porous_id
    sm.array()[fluid_mask] = fluid_id

    fluid_restr = generate_subdomain_restriction(mesh, sm, fluid_id)
    porous_restr = generate_subdomain_restriction(mesh, sm, porous_id)
    fluid_restr._write(f"meshes/{mesh_name}/{mesh_name}_fluid.rtc.xdmf")
    porous_restr._write(f"meshes/{mesh_name}/{mesh_name}_porous.rtc.xdmf")

if __name__=="__main__":
    import sys
    mesh_name = sys.argv[1]
    with open(f"config_files/{mesh_name}.yml") as f:
        mesh_config = yaml.load(f, Loader=yaml.FullLoader)

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_ids = sum(mesh_config["fluid_comp_ids"].values(), [])

    # read in mesh, mark pvs and save again
    mesh = pv.read(f"meshes/{mesh_name}/mesh.xdmf")
    pvs = pv.read(f"meshes/{mesh_name}/surfaces/pvs.stl").extract_geometry()
    lumen = pv.read(f"meshes/{mesh_name}/surfaces/lumen.stl").extract_geometry()

    mesh = mark_pvs(mesh, pvs)
    mesh.compute_implicit_distance(lumen, inplace=True)

    if mesh_config.get("homogenize_endfeet", False):
        pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
        outer_cell_id = mesh_config["porous_cell_ids"]["outer_cell"]
        outer = np.isin(mesh["subdomains"], pvs_ids) == False
        mesh["subdomains"][outer] = outer_cell_id

    mesh.points *= 1e-9
    pv.save_meshio(f"meshes/{mesh_name}/{mesh_name}.xdmf", mesh)

    mesh = Mesh()
    with XDMFFile(f"meshes/{mesh_name}/{mesh_name}.xdmf") as mf:
        mf.read(mesh)
        sm = MeshFunction("size_t", mesh, 3, 0)
        mf.read(sm, "subdomains")
        sm.rename("subdomains","")

    bm = mark_boundaries(mesh, sm, cell_ids, eps=0.004)

    sm.rename("subdomains","")
    bm.rename("boundaries","")

    with XDMFFile(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf") as outfile:
        outfile.write(bm)

    with XDMFFile(f"meshes/{mesh_name}/{mesh_name}.xdmf") as outfile:
        outfile.write(sm,)

    generate_subdomains(mesh, sm, cell_ids, fluid_ids, mesh_name)