import pyvista as pv
import sys
from plotting.endfeet_buffering_visualisation_k3d import read_results
import numpy as np
import yaml
from multiprocessing import Pool
from functools import partial
import time
import os

num_cpus = len(os.sched_getaffinity(0))

if __name__ == "__main__":
    start = time.time()
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    mesh, bm, mesh_config, config = read_results(sim_name, mesh_name)
    astro_ids = mesh_config["porous_cell_ids"]["astrocyte"]
    ecs_id = mesh_config["fluid_comp_ids"]["ecs"]
    pvs_id = mesh_config["fluid_comp_ids"]["pvs"]
    ecs_outer_id = mesh_config["facet_ids"]["ecs_outer_id"]
    outer_cell_id = mesh_config["porous_cell_ids"].get("outer_cell", [])


    num_steps = config["num_steps"]
    dt = config["T"] / num_steps
    times = np.linspace(0, num_steps*dt, num_steps)
    results = dict()
    num_cycles = config["T"] * config["f"]
    steps_per_cycle = num_steps / num_cycles
    peak_time_idx = int((num_cycles - 0.25) * steps_per_cycle) #assuming peak at 1/4 of the last cycle
    N = 10000

    ecs = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], ecs_id ))
    pvs = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], pvs_id ))
    ef = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], astro_ids + outer_cell_id ))
    ecs_outer = mesh.extract_points(bm["boundaries"]==ecs_outer_id)

    subdomains = {"ecs":ecs, "pvs":pvs, "ef":ef}
    intra_quantities = ["pP_i", "vm", "d", "darcy"]
    extra_quantities = ["pP_e", "vm", "d", "darcy"]
    quantities = {"ecs":extra_quantities, "pvs":extra_quantities,
                   "ef":intra_quantities}
    #from IPython import embed; embed()
    for subdn, subd in subdomains.items():
        if subd.n_cells==0:
            continue
        subd = subd.point_data_to_cell_data()
        subd = subd.compute_cell_sizes()
        v = abs(subd["Volume"])
        for q in quantities[subdn]:
            data = subd[f"{q}_{peak_time_idx}"]
            if data.ndim ==2:
                data = np.linalg.norm(data, axis=1)
            sample = np.random.choice(data,
                                      size=N, p=v / v.sum())
            results[f"{subdn}_{q}_samples"] = sample

    for k,v in results.items():
        results[k] = np.array(v).tolist()

    with open(f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_peak_distr.yml", 'w') as file:
        yaml.dump(results, file)