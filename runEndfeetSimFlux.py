import numpy as np
import argparse
import yaml
from petsc4py import PETSc
from fenics import *
from collections import defaultdict
from biotbiot.utils import (
    meshFunction_to_DG,
    get_lumen_centerline_direction,
    InterfaceExpression,
    update_material_params
)
from biotbiot.biotBiotFluxSolver import solve as solve_model

print = PETSc.Sys.Print
set_log_level(20)
parameters["ghost_mode"] = "shared_facet"
parameters["mesh_partitioner"] = "ParMETIS"  # ParMETIS

PETScOptions.set("mat_mumps_use_omp_threads", 8)
PETScOptions.set("mat_mumps_icntl_35", True)  # set BLR
PETScOptions.set("mat_mumps_cntl_7", 1e-8)  # set BLR relaxation
PETScOptions.set("mat_mumps_icntl_4", 3)  # verbosity
#PETScOptions.set("mat_mumps_icntl_24", 1)  # detect null pivot rows
PETScOptions.set("mat_mumps_icntl_22", 0)  # out of core
PETScOptions.set("mat_mumps_icntl_28", 2)  # parallel ordering
PETScOptions.set("mat_mumps_icntl_29", 2)  # parmetis
PETScOptions.set("mat_mumps_icntl_14", 10)  # max memory increase in %

R = 8.314  # e18
temp = 310.15
v_w = 18 * 1e-6  # 1.8e+22

porous_id = 1
fluid_id = 2

def generate_bm_marker(mesh, mvc, aqp_membrane_id, interf_id):
    bm = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    bm_aqp = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    bm.array()[bm.array() > 1e12] = 0
    bm_aqp.array()[bm_aqp.array() > 1e12] = 0
    bm.array()[bm.array()[:] == aqp_membrane_id] = interf_id
    return bm, bm_aqp

def generate_biot_biot_bcs(mesh, mvc, facet_ids, config, center_dir):
    pvs_inlet_id = facet_ids["pvs_inlet_id"]
    pvs_outlet_id = facet_ids["pvs_outlet_id"]
    endfeet_neck_outer_id = facet_ids["endfeet_neck_outer_id"]
    endfeet_axial_id = facet_ids["endfeet_axial_id"]
    arterial_wall_id = facet_ids["arterial_wall_id"]
    interf_id = facet_ids["interf_id"]
    aqp_membrane_id = facet_ids["aqp_membrane_id"]
    cells_outer_id = facet_ids["cells_outer_id"]
    cells_axial_id = facet_ids["cells_axial_id"]
    ecs_axial_id = facet_ids["ecs_axial_id"]
    ecs_outer_id = facet_ids["ecs_outer_id"]

    f = config["f"]

    bm, bm_aqp = generate_bm_marker(mesh, mvc, aqp_membrane_id, interf_id)

    mat_params = config["material_parameters"]
    P_f_apq = mat_params["membrane"]["P_f_aqp"]
    P_f_wo_apq = mat_params["membrane"]["P_f_wo_aqp"]

    Lp_aqp = P_f_apq * v_w / (R * temp)
    Lp_wo_aqp = P_f_wo_apq * v_w / (R * temp)

    marker_value_dict = defaultdict(
        float, {aqp_membrane_id: Lp_aqp, interf_id: Lp_wo_aqp}
    )
    L_p = InterfaceExpression(mesh, bm_aqp, marker_value_dict, degree=1)
    mat_params["L_p"] = L_p

    V = VectorFunctionSpace(mesh, "CG", 1)
    vessel_disp = Function(V)
    vessel_flow = Function(V)

    c_dir =  center_dir.vector()[:]
    A = config["vessel_pulsation"]

    disp_velo_ratio = config.get("disp_velo_ratio", 0)

    def update(t, sol):
        vessel_disp.vector()[:] = - c_dir * A * np.sin(2 * np.pi * f * t)
        vessel_flow.vector()[:] = - c_dir * A * np.cos(2*np.pi*f*t) * 2 * np.pi * f * disp_velo_ratio
    
    if disp_velo_ratio ==0:
        print("Using lagrangian framework...")
    zero_vec = Constant((0, 0, 0))
    boundary_conditions = [
        # wall movement
        (arterial_wall_id, 0, vessel_disp, bm),
        (arterial_wall_id, 1, vessel_flow, bm),
        # tangential displacement
        (pvs_inlet_id, (0, 1), 0, bm),
        (pvs_outlet_id, (0, 1), 0, bm),
        (ecs_axial_id, (0, 1), 0, bm),
        (cells_axial_id, (0, 1), 0, bm),
        (pvs_inlet_id, 1, zero_vec, bm),
        # no axial flow
        (ecs_axial_id, 1, zero_vec, bm),
        (cells_axial_id, 1, zero_vec, bm),
    ]
    if config["periodic"]:
        boundary_conditions.append((pvs_outlet_id, 1, zero_vec, bm))

    kappa_b = mat_params["boundary"]["kappa"]
    mu_f = mat_params["extracellular"]["mu_f"]
    p_far = mat_params["boundary"]["p_far_field"]
    l = mat_params["boundary"]["dist_far_field"]
    E = mat_params["boundary"]["E"]

    ds = Measure("ds", mesh, subdomain_data=bm)
    dS = Measure("dS", mesh, subdomain_data=bm)
    dSaqp = Measure("dS", mesh, subdomain_data=bm_aqp)
    K = kappa_b / mu_f

    n = FacetNormal(mesh)

    robin_boundaries =  [(ds(ecs_outer_id), 1, lambda q,v: inner(q,n)*inner(v,n)*l/K, lambda q: inner(q,n*Constant(p_far))),
                         (ds(ecs_outer_id), 0, lambda d,w: Constant(E/l)*inner(d,w), lambda d: inner(d,zero_vec)),
                         (ds(cells_outer_id), 1, lambda q,v: inner(q,n)*inner(v,n)*l/K, lambda q: inner(q,n*Constant(p_far))),
                         (ds(cells_outer_id), 0, lambda d,w: Constant(E/l)*inner(d,w), lambda d: inner(d,zero_vec)),
                        ]
    
    dp = config.get("dp", 0)
    if dp:
        dp = InterfaceExpression(mesh, bm_aqp, {aqp_membrane_id:dp, interf_id:0}, degree=1)
        robin_boundaries.append((dS(interf_id), 1, lambda q,v:0, lambda v: inner(avg(v), n("+")*dp)))
        assert assemble(dp*dS(interf_id)) == assemble(dp*dSaqp(aqp_membrane_id))

    tde = [update]
    return boundary_conditions, robin_boundaries, bm, tde


def read_mesh(mesh_name, mesh_config):
    mesh = Mesh()
    infile = XDMFFile(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    infile.read(mesh)

    gdim = mesh.geometric_dimension()
    sm = MeshFunction("size_t", mesh, gdim)
    infile.read(sm, "subdomains")
    sublabels = MeshFunction("size_t", mesh, gdim)
    infile.read(sublabels, "subdomains")
    infile.close()

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_ids = sum(mesh_config["fluid_comp_ids"].values(), [])
    porous_mask = np.isin(sm.array(), cell_ids)
    fluid_mask = np.isin(sm.array(), fluid_ids)

    sm.array()[porous_mask] = porous_id
    sm.array()[fluid_mask] = fluid_id

    mvc = MeshValueCollection("size_t", mesh, gdim - 1)

    facet_infile = XDMFFile(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf")
    facet_infile.read(mvc)
    facet_infile.close()

    lumen_dir = get_lumen_centerline_direction(mesh, mesh_name)
    return mesh, sm, sublabels, mvc, lumen_dir


def run_simulation(config_name, mesh_name):

    with open(f"config_files/{config_name}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    facet_ids = mesh_config["facet_ids"]
    T = config["T"]
    num_steps = config["num_steps"]
    f = config["f"]

    mat_params = config["material_parameters"]
    if "pvs" not in mat_params.keys():
        mat_params["pvs"] = mat_params["extracellular"]
        print("no PVS material parameters found, using ECS values")
    update_material_params(mat_params)

    print(mat_params["intracellular"])
    print(mat_params["extracellular"])
    print(mat_params["pvs"])

    print("reading mesh...")
    mesh, sm, sublabels, mvc, lumen_dir = read_mesh(mesh_name, mesh_config)
    print("read mesh.")

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
    ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]

    for param in ["K", "alpha", "lmbda", "mu_s", "c"]:
        param_map = dict(
            zip(
                cell_ids + pvs_ids + ecs_ids,
                [mat_params["intracellular"][param]] * len(cell_ids)
                + [mat_params["pvs"][param]] * len(pvs_ids)
                + [mat_params["extracellular"][param]] * len(ecs_ids),
            )
        )
        dg_param_func = meshFunction_to_DG(sublabels, param_map)
        mat_params[param] = dg_param_func
    interf_id = facet_ids["interf_id"]

    (
        boundary_conditions,
        robin_conditions,
        bm,
        tde,
    ) = generate_biot_biot_bcs(mesh, mvc, facet_ids, config, lumen_dir)

    solve_model(
        mesh,
        T,
        num_steps,
        mat_params,
        bm,
        sm,
        boundary_conditions,
        f"results/{mesh_name}_{config_name}/{mesh_name}_{config_name}",
        time_dep_expr=tde,
        robin_boundaries=robin_conditions,
        interface_id=interf_id,
        degree=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        metavar="config.yml",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-m",
        metavar="mesh.xdmf",
        help="path to mesh file",
        type=str,
    )
    conf_arg = vars(parser.parse_args())
    config_file_path = conf_arg["c"]
    mesh_file_path = conf_arg["m"]
    run_simulation(config_file_path, mesh_file_path)
