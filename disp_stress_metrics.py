
from fenics import *
import numpy as np
from biotbiot.biotBiotFluxWeakForm import create_function_spaces
import os, sys
import yaml
from tqdm import tqdm

parameters["ghost_mode"] = "shared_facet"
parameters["mesh_partitioner"] = "ParMETIS"

porous_id  = 1
fluid_id  = 2

def compute_disp_metrics(name, mesh_name):

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
    outer_cell_ids = mesh_config["porous_cell_ids"]["outer_cell"]

    outer_cell_mask = np.isin(sm.array(), outer_cell_ids)

    sm.array()[outer_cell_mask] = min(outer_cell_ids)
    outer_cell_ids = [min(outer_cell_ids)]

    mvc = MeshValueCollection("size_t", mesh, gdim - 1)

    facet_infile = XDMFFile(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf")
    facet_infile.read(mvc)
    facet_infile.close()
    bm = cpp.mesh.MeshFunctionSizet(mesh, mvc)

    bm.array()[bm.array()>1e12] = 0

    facet_infile.close()

    f = config["f"]
    T = config["T"]
    num_steps = config["num_steps"]
    dt = T/num_steps

    dS = Measure("dS", mesh, subdomain_data=bm)
    dx = Measure("dx", mesh, subdomain_data=sm)

    infile = HDF5File(mesh.mpi_comm(), f"results/{mesh_name}_{name}/{mesh_name}_{name}.hdf", "r")
    H = create_function_spaces(mesh)
    V = H.sub(0).collapse()
    DG0 = FunctionSpace(mesh, "DG", 0)

    variables = {"d":V, "vm":DG0}
    results = {n:[] for n in variables.keys()}

    print("reading in data...")

    for n, space in variables.items():
        for i in tqdm(range(num_steps)):
            f = Function(space)
            infile.read(f,f"{n}/vector_{i}")
            results[n].append(f)
    infile.close()

    print("computing metrics...")

    # compute mean displacement
    def mean_disp(u, ids):
        volumes = []
        mean_flows = []
        for subd in ids:
            volumes.append(assemble(1*dx(subd)))
            mean_flows.append( assemble(sqrt( inner(u,u) )*dx(subd)) )
        volumes = np.array(volumes)
        mean_flows = np.array(mean_flows)
        mean_flow = mean_flows.sum() / volumes.sum()
        return mean_flow

    D = ["pvs", "ecs", "ef", "oc"]
    pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
    ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
    ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]

    D_ids = [pvs_ids, ecs_ids, ef_ids, outer_cell_ids]
    avg_disp = {f"{dom}_displacement":[] for dom in D}
    for dom, ids in zip(D,D_ids):
        for d in results["d"]:
            avg_disp[f"{dom}_displacement"].append(mean_disp(d, ids))
        
    avg_mises = {f"{dom}_von_mises":[] for dom in D}
    for dom, ids in zip(D,D_ids):
        for d in results["vm"]:
            avg_mises[f"{dom}_von_mises"].append(mean_disp(d, ids))

    data = avg_disp | avg_mises

    ecs_id = ecs_ids[0]
    pvs_id = pvs_ids[0]
    mesh.scale(1e6)
    for d in results["d"]:
        d.vector()[:] *= 1e6

    for dom, did in zip(["ecs_volume", "pvs_volume"] + [f"astrocyte_volume_{i}" for i in range(len(ef_ids))] + ["oc_volume"], 
                        [ecs_id, pvs_id] + ef_ids + outer_cell_ids):
        vol_div = []
        for d in results["d"]:
            vol_div.append(assemble( (1 + div(d))*dx(did)))
        data[f"{dom}"] = vol_div

    data["ef_volume"] = [sum([data[f"astrocyte_volume_{i}"][k] for i in range(len(ef_ids))]) for k in range(num_steps)]
    
    return data

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    results = compute_disp_metrics(sim_name, mesh_name)

    for k,v in results.items():
        results[k] = np.array(v).tolist()
    with open(f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_disp_stress_metrics.yml", 'w') as file:
        yaml.dump(results, file)