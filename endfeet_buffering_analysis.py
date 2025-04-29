
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from biotbiot.biotBiotFluxWeakForm import create_function_spaces, create_measures
plt.style.use("bmh")
matplotlib.rc('mathtext', default = "regular")
import os, sys
import yaml
from tqdm import tqdm
parameters["ghost_mode"] = "shared_facet"

porous_id  = 1
fluid_id  = 2

def analyse_endfeet_results(name, mesh_name):
    data = dict()

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    facet_ids = mesh_config["facet_ids"]

    with open(f"config_files/{name}.yml") as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

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

    f = config["f"]
    T = config["T"]
    num_steps = 5 #config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps)

    infile = HDF5File(mesh.mpi_comm(), f"results/{mesh_name}_{name}/{mesh_name}_{name}.hdf", "r")
    H = create_function_spaces(mesh)

    variables = {"d": H.sub(0).collapse(), "darcy":H.sub(1).collapse(), "p":H.sub(2).collapse()}

    results = {n:[] for n in variables.keys()}

    for n, space in variables.items():
        for i in tqdm(range(num_steps)):
            f = Function(space)
            infile.read(f,f"{n}/vector_{i}")
            results[n].append(f)
    infile.close()

    data = dict()

    # compute volume changes
    mesh.scale(1e6)
    for d in results["d"]:
        d.vector()[:] *= 1e6

    dx = Measure("dx", mesh, subdomain_data=sm2)

    vol_div = []
    pvs_id = mesh_config["fluid_comp_ids"]["pvs"][0]
    astro_ids = mesh_config["porous_cell_ids"]["astrocyte"]
    astro_ids += mesh_config["porous_cell_ids"].get("outer_cell", [])
    ecs_id = mesh_config["fluid_comp_ids"]["ecs"][0]

    for dom, did in zip(["ecs_volume", "pvs_volume"] + [f"astrocyte_volume_{i}" for i in range(len(astro_ids))],
                        [ecs_id, pvs_id] + astro_ids):
        for d in results["d"]:
            vol_div.append(assemble(div(d)*dx(did)))

        vol_ale = []
        initial_vol = assemble(Constant(1)*dx(did))
        for d in results["d"]:
            ALE.move(mesh, d)
            vol_ale.append(assemble(Constant(1)*dx(did)) - initial_vol)
            d.vector()[:] *= -1
            ALE.move(mesh, d)
            d.vector()[:] *= -1
        data[f"{dom}"] = vol_ale

    data["ef_volume"] = [sum([data[f"astrocyte_volume_{i}"][k] for i in range(len(astro_ids))]) for k in range(num_steps)]
    # scale back
    mesh.scale(1e-6)

    # compute outflows
    n = FacetNormal(mesh)
    outflows = {"pvs_inlet_id":[],
                "pvs_outlet_id":[],
                "ecs_outer_id":[],
                }

    ds = Measure("ds", mesh, subdomain_data=bm)
    for boundary, outflow_ts in outflows.items():
        for q in results["darcy"]:
            boundary_id = facet_ids[boundary]
            flow = assemble(dot(q,n)*ds(boundary_id))
            outflow_ts.append(flow)
        outflows[boundary] = np.array(outflow_ts)

    # cellular outflow

    cell_outflows = {"endfeet_neck_outer_id":[]}

    for boundary, outflow_ts in cell_outflows.items():
        for q in results["darcy"]:
            boundary_id = facet_ids[boundary]
            flow = assemble(inner(q, n)*ds(boundary_id))
            outflow_ts.append(flow)
        cell_outflows[boundary] = np.array(outflow_ts)

    # flow across the membrane:
    n = FacetNormal(mesh)
    membrane_flow = {"aqp_membrane_id":[],
                    "interf_id":[],
                }

    dS = Measure("dS", mesh, subdomain_data=bm)
    dx = Measure("dx", mesh, subdomain_data=sm)
    for boundary, flow_ts in membrane_flow.items():
        for q in results["darcy"]:
            boundary_id = facet_ids[boundary]
            flow = assemble(inner(q("+"),n("+"))*dS(boundary_id) + Constant(0)*inner(q,q)*dx)
            flow_ts.append(flow)
        membrane_flow[boundary] = np.array(flow_ts)


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
            mean_flows.append(assemble(sqrt(inner(u,u))*dx(subd)))
        volumes = np.array(volumes)
        mean_flows = np.array(mean_flows)
        mean_flow = mean_flows.sum() / volumes.sum()
        return mean_flow

    pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
    ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
    ics_ids = sum(mesh_config["porous_cell_ids"].values(),[])

    mean_ecs_flow = [mean_flow(u, ecs_ids) for u in results["darcy"]]
    mean_pvs_flow = [mean_flow(u, pvs_ids) for u in results["darcy"]]
    mean_ics_flow = [mean_flow(u, ics_ids) for u in results["darcy"]]

    data["mean_ecs_flow"] = mean_ecs_flow
    data["mean_pvs_flow"] = mean_pvs_flow
    data["mean_ics_flow"] = mean_ics_flow

    # pressure values
    ecs_outer_id = facet_ids["ecs_outer_id"]
    interf_id = facet_ids["aqp_membrane_id"]
    ecs_outer_surface = assemble(1*ds(ecs_outer_id))
    interf_surface = assemble(1*dS(interf_id))
    ecs_outer_mean_pressure = []
    ecs_interf_mean_pressure = []
    
    for p in results["pP_e"]:
        p_mean = assemble(p*ds(ecs_outer_id))
        ecs_outer_mean_pressure.append(p_mean)
        p_mean = assemble(p*dS(interf_id))
        ecs_interf_mean_pressure.append(p_mean)

    if ecs_outer_surface > 0:
        ecs_outer_mean_pressure = np.array(ecs_outer_mean_pressure) / ecs_outer_surface
    if interf_surface > 0:
        ecs_interf_mean_pressure = np.array(ecs_interf_mean_pressure) / interf_surface

    endfeet_neck_outer_id = facet_ids["endfeet_neck_outer_id"]
    endfeet_neck_outer_surface = assemble(1*ds(endfeet_neck_outer_id))
    astro_outer_mean_pressure = []
    astro_interf_mean_pressure = []

    for p in results["pP_i"]:
        p_mean = assemble(p*ds(endfeet_neck_outer_id))
        astro_outer_mean_pressure.append(p_mean)
        p_mean = assemble(p*dS(interf_id))
        astro_interf_mean_pressure.append(p_mean)

    if endfeet_neck_outer_surface > 0:
        astro_outer_mean_pressure = np.array(astro_outer_mean_pressure)
        astro_outer_mean_pressure /= endfeet_neck_outer_surface
    if interf_surface > 0:
        astro_interf_mean_pressure = np.array(astro_interf_mean_pressure)
        astro_interf_mean_pressure /= interf_surface
        
    data["aqp_membrane_flow"] = membrane_flow["aqp_membrane_id"]
    data["non_aqp_membrane_flow"] = membrane_flow["interf_id"]
    data["mean_y_velocity"] = mean_vel
    data["inlet_flow"] = outflows["pvs_inlet_id"]
    data["outlet_flow"] = outflows["pvs_outlet_id"]
    data["ecs_flow"] = outflows["ecs_outer_id"]
    data["endfeet_neck_flow"] = cell_outflows["endfeet_neck_outer_id"]
    data["ecs_outer_mean_pressure"] = ecs_outer_mean_pressure
    data["astro_outer_mean_pressure"] = astro_outer_mean_pressure
    data["astro_interf_mean_pressure"] = astro_interf_mean_pressure
    data["ecs_interf_mean_pressure"] = ecs_interf_mean_pressure

    data["interf_surface"] = interf_surface
    data["ecs_outer_surface"] = ecs_outer_surface
    data["endfeet_neck_outer_surface"] = endfeet_neck_outer_surface
    return data


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    results = analyse_endfeet_results(sim_name, mesh_name)

    for k,v in results.items():
        results[k] = np.array(v).tolist()

    with open(f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_fenics.yml", 'w') as file:
        yaml.dump(results, file)