from petsc4py import PETSc
from fenics import *
import numpy as np
from biotbiot.biotBiotFluxWeakForm import create_function_spaces
import os, sys
import yaml
from meshing.mark_arteriole_boundaries import mark_internal_interface
from tqdm import tqdm

print = PETSc.Sys.Print
parameters["ghost_mode"] = "shared_facet"
parameters["mesh_partitioner"] = "ParMETIS"

porous_id  = 1
fluid_id  = 2

def compute_flow_metrics(name, mesh_name):
    data = dict()

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    facet_ids = mesh_config["facet_ids"]

    with open(f"config_files/{name}.yml") as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

    print("reading the mesh...")
    mesh = Mesh()
    infile =  XDMFFile(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    infile.read(mesh)

    gdim = mesh.geometric_dimension()
    sm = MeshFunction("size_t", mesh, gdim)
    infile.read(sm, "subdomains")
    sm2 = MeshFunction("size_t", mesh, gdim)
    infile.read(sm2, "subdomains")
    infile.close()

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_comp_ids = sum(mesh_config["fluid_comp_ids"].values(), [])

    fluid_mask = np.isin(sm.array(), fluid_comp_ids)
    porous_mask = np.isin(sm.array(), cell_ids)

    sm.array()[porous_mask] = porous_id
    sm.array()[fluid_mask] = fluid_id

    mvc = MeshValueCollection("size_t", mesh, gdim - 1)

    facet_infile = XDMFFile(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf")
    facet_infile.read(mvc)
    facet_infile.close()
    bm = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    bm.array()[bm.array()>1e12] = 0

    facet_infile.close()
    print("mesh read in.")

    print("marking endfeet gaps...")
    endfeet_gap_id = 101
    facet_ids["endfeet_gap_id"] = endfeet_gap_id
    mark_internal_interface(mesh, sm2, bm, mesh_config["fluid_comp_ids"]["pvs"],
                            mesh_config["fluid_comp_ids"]["ecs"], endfeet_gap_id)
    print("marked endfeet gaps")
    f = config["f"]
    T = config["T"]
    num_steps = config["num_steps"]
    dt = T/num_steps

    infile = HDF5File(mesh.mpi_comm(), f"results/{mesh_name}_{name}/{mesh_name}_{name}.hdf", "r")

    H = create_function_spaces(mesh)

    variables = {"darcy": H.sub(1).collapse(),
                 "p":H.sub(2).collapse()}
    
    intracell_p = "p"
    extracell_p = "p"
    intra_restr = "+"
    extra_restr = "-"

    results = {n:[] for n in variables.keys()}

    for n, space in variables.items():
        print(f"reading {n}")
        for i in tqdm(range(num_steps)):
            f = Function(space)
            infile.read(f,f"{n}/vector_{i}")
            results[n].append(f)
    infile.close()

    # compute outflows
    n = FacetNormal(mesh)
    outflows = {"pvs_inlet_id":[],
                "pvs_outlet_id":[],
                "ecs_outer_id":[],
                }

    print("computing outflows...")
    ds = Measure("ds", mesh, subdomain_data=bm)
    for boundary, outflow_ts in outflows.items():
        for q in results["darcy"]:
            boundary_id = facet_ids[boundary]
            flow = assemble(dot(q,n)*ds(boundary_id))
            outflow_ts.append(flow)
        outflows[boundary] = np.array(outflow_ts)


    # cellular outflow
    cell_outflows = {"endfeet_neck_outer_id":[]}
    print("computing cell outflows")
    for boundary, outflow_ts in cell_outflows.items():
        for q in tqdm(results["darcy"]):
            boundary_id = facet_ids[boundary]
            flow = assemble(inner(q, n)*ds(boundary_id))
            outflow_ts.append(flow)
        cell_outflows[boundary] = np.array(outflow_ts)

    # flow across the membrane:

    n = FacetNormal(mesh)
    membrane_flow = {"aqp_membrane_id":[],
                    "interf_id":[],
                    "endfeet_gap_id":[]
                    }
    R = 8.314
    temp = 310.15
    v_w = 18 * 1e-6
    P_f_apq = config["material_parameters"]["membrane"]["P_f_aqp"]
    P_f_wo_apq = config["material_parameters"]["membrane"]["P_f_wo_aqp"]
    Lp_aqp = P_f_apq * v_w / (R * temp)
    Lp_wo_aqp = P_f_wo_apq * v_w / (R * temp)
    L_p = dict(aqp_membrane_id=Lp_aqp, interf_id=Lp_wo_aqp)

    print("computing membrane flow")
    dS = Measure("dS", mesh, subdomain_data=bm)
    dx = Measure("dx", mesh, subdomain_data=sm2)
    for boundary, flow_ts in membrane_flow.items():
        boundary_id = facet_ids[boundary]

        for pi, pe, q in zip(results[intracell_p], results[extracell_p], results["darcy"]):

            flow = assemble(inner(q(intra_restr),n("+"))*dS(boundary_id) + Constant(0)*inner(q,q)*dx)
            area = assemble(1*dS(boundary_id))
            if area > 0 and boundary in L_p.keys():
                dp = assemble((pe(extra_restr) - pi(intra_restr))*dS(boundary_id) + Constant(0)*inner(pi,pe)*dx) /area
                mean_flow = dp* L_p[boundary] * assemble(1*dS(boundary_id))
            flow_ts.append(flow)
        membrane_flow[boundary] = np.array(flow_ts)

    # surface areas
    print("computing surface areas")
    surfaces = list(outflows) + list(cell_outflows) + list(membrane_flow)
    for s in surfaces:
        boundary_id = facet_ids[s]
        data[f"{s}_area"] = assemble(1*ds(boundary_id) + 1* dS(boundary_id))
    # PVS velocity
    dx = Measure("dx", mesh, subdomain_data=sm2)

    mean_vel = []
    pvs_id = mesh_config["fluid_comp_ids"]["pvs"][0]


    pvs_vol = assemble(1e6*dx(pvs_id))
    for u in results["darcy"]:
        u_mean_y = assemble(inner(u * 1e6, as_vector([0,1,0]))*dx(pvs_id)) / pvs_vol
        mean_vel.append(u_mean_y)
    mean_vel = np.array(mean_vel)

    # mean flow velocities:

    def mean_flow(u, ids):
        volumes = []
        mean_flows = []
        for subd in ids:
            volumes.append(assemble(1*dx(subd)))
            mean_flows.append( assemble(sqrt( inner(u,u) )*dx(subd)) )
        volumes = np.array(volumes)
        mean_flows = np.array(mean_flows)
        mean_flow = mean_flows.sum() / volumes.sum()
        return mean_flow

    pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
    ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
    ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]
    oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]

    print("computing mean velocities")
    mean_ecs_flow = [mean_flow(u, ecs_ids) for u in tqdm(results["darcy"])]
    mean_pvs_flow = [mean_flow(u, pvs_ids) for u in tqdm(results["darcy"])]
    mean_ef_flow = [mean_flow(u, ef_ids) for u in tqdm(results["darcy"])]
    mean_oc_flow = [mean_flow(u, oc_ids) for u in tqdm(results["darcy"])]

    data["mean_ecs_flow"] = mean_ecs_flow
    data["mean_pvs_flow"] = mean_pvs_flow
    data["mean_ef_flow"] = mean_ef_flow
    data["mean_oc_flow"] = mean_oc_flow

    # pressure values
    ecs_outer_id = facet_ids["ecs_outer_id"]
    interf_id = facet_ids["aqp_membrane_id"]
    ecs_outer_surface = assemble(1*ds(ecs_outer_id))
    interf_surface = assemble(1*dS(interf_id))
    ecs_outer_mean_pressure = []
    ecs_interf_mean_pressure = []
    
    for p in results[extracell_p]:
        p_mean = assemble(p*ds(ecs_outer_id) )
        ecs_outer_mean_pressure.append(p_mean)
        p_mean = assemble(p(extra_restr)*dS(interf_id) + Constant(0)*p*dx)
        ecs_interf_mean_pressure.append(p_mean)

    if ecs_outer_surface > 0:
        ecs_outer_mean_pressure = np.array(ecs_outer_mean_pressure) / ecs_outer_surface
    if interf_surface > 0:
        ecs_interf_mean_pressure = np.array(ecs_interf_mean_pressure) / interf_surface

    endfeet_neck_outer_id = facet_ids["endfeet_neck_outer_id"]
    endfeet_neck_outer_surface = assemble(1*ds(endfeet_neck_outer_id))
    astro_outer_mean_pressure = []
    astro_interf_mean_pressure = []

    for p in results[intracell_p]:
        p_mean = assemble(p*ds(endfeet_neck_outer_id))
        astro_outer_mean_pressure.append(p_mean)
        p_mean = assemble(p(intra_restr)*dS(interf_id)+ Constant(0)*p*dx)
        astro_interf_mean_pressure.append(p_mean)

    if endfeet_neck_outer_surface > 0:
        astro_outer_mean_pressure = np.array(astro_outer_mean_pressure)
        astro_outer_mean_pressure /= endfeet_neck_outer_surface
    if interf_surface > 0:
        astro_interf_mean_pressure = np.array(astro_interf_mean_pressure)
        astro_interf_mean_pressure /= interf_surface

    
    membrane_volumetric_flow_rate = membrane_flow["aqp_membrane_id"] + membrane_flow["interf_id"]
    endfeet_volume_change = membrane_volumetric_flow_rate  + cell_outflows["endfeet_neck_outer_id"]
    peak_endfoot_volume_change = (endfeet_volume_change*dt).cumsum().max()

    # flow 
    data["aqp_membrane_flow"] = membrane_flow["aqp_membrane_id"]
    data["non_aqp_membrane_flow"] = membrane_flow["interf_id"]
    data["endfeet_gap_flow"] = membrane_flow["endfeet_gap_id"]
    data["inlet_flow"] = outflows["pvs_inlet_id"]
    data["outlet_flow"] = outflows["pvs_outlet_id"]
    data["ecs_flow"] = outflows["ecs_outer_id"]
    data["endfeet_neck_flow"] = cell_outflows["endfeet_neck_outer_id"]
    data["membrane_volumetric_flow_rate"] = membrane_volumetric_flow_rate
    data["endfeet_volume_change"] = endfeet_volume_change
    data["peak_endfoot_volume_change"] = peak_endfoot_volume_change
    data["mean_y_velocity"] = mean_vel

    # mean pressure
    data["ecs_outer_mean_pressure"] = ecs_outer_mean_pressure
    data["astro_outer_mean_pressure"] = astro_outer_mean_pressure
    data["astro_interf_mean_pressure"] = astro_interf_mean_pressure
    data["ecs_interf_mean_pressure"] = ecs_interf_mean_pressure

    data["endfeet_volume"] = assemble(1*dx(porous_id))

    return data


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    results = compute_flow_metrics(sim_name, mesh_name)

    for k,v in results.items():
        results[k] = np.array(v).tolist()

    with open(f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_flow_metrics.yml", 'w') as file:
        yaml.dump(results, file)