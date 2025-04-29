import pyvista as pv
import sys
from plotting.endfeet_buffering_visualisation_k3d import read_results
import numpy as np
import yaml
from IPython import embed


def get_axial_mean(mesh, times, scalar, domain_ids=None, axis="y", n_slices=50):
    if domain_ids:
        mesh = mesh.extract_cells(np.isin(mesh["subdomains"], domain_ids))
    length = mesh.bounds[3] - mesh.bounds[2]
    slices = mesh.slice_along_axis(n=n_slices, axis=axis, tolerance=0.05*length)
    #from IPython import embed; embed()
    vals = []
    for sl in slices:
        vals.append(get_range(sl, times, scalar)[1])
    coords = [sl.bounds[2] for sl in slices]
    return np.array(vals), coords

def get_range(mesh, times, scalar):
    min_q, mean_q, max_q = [] , [], []
    sized = mesh.compute_cell_sizes()
    mesh_cell = mesh.point_data_to_cell_data()
    vol = np.abs(sized["Volume"])
    total_vol  = vol.sum()
    if np.isclose(total_vol, 0.0):
        vol = np.abs(sized["Area"])
        total_vol  = vol.sum()
    for i, t in enumerate(times):
        max_q.append(mesh[f"{scalar}_{i}"].max())
        mean_q.append((mesh_cell[f"{scalar}_{i}"] * vol).sum() / total_vol)
        min_q.append(mesh[f"{scalar}_{i}"].min())
    return np.array(min_q), np.array(mean_q), np.array(max_q)


def evaluate_scalar(mesh, time, scalar, ids):
    domain = mesh.extract_cells(np.isin(mesh["subdomains"], ids))
    dom = pv.UnstructuredGrid()
    dom.copy_structure(domain)
    for i,t in enumerate(time):
        dom[f"{scalar}_{i}"] = domain[f"{scalar}_{i}"]
    return get_range(dom, time, scalar)


if __name__ == "__main__":
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    
    mesh, bm, mesh_config, config = read_results(sim_name, mesh_name)
    astro_ids = mesh_config["porous_cell_ids"]["astrocyte"]
    oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]

    ecs_id = mesh_config["fluid_comp_ids"]["ecs"]
    pvs_id = mesh_config["fluid_comp_ids"]["pvs"]

    intra_p = "p"
    extra_p = "p"

    num_steps = config["num_steps"]
    dt = config["T"] / num_steps
    times = np.linspace(0, config["T"], num_steps)

    ef_vm = evaluate_scalar(mesh, times, "vm", astro_ids)
    pvs_vm = evaluate_scalar(mesh, times, "vm", pvs_id)
    oc_vm = evaluate_scalar(mesh, times, "vm", oc_ids)

    ef_p = evaluate_scalar(mesh, times, intra_p, astro_ids)
    oc_p = evaluate_scalar(mesh, times, intra_p, oc_ids)
    pvs_p = evaluate_scalar(mesh, times, extra_p, pvs_id)

    #ef_phi = evaluate_scalar(mesh, times, "phi_i", cell_ids)
    #pvs_phi = evaluate_scalar(mesh, times, "phi_e", pvs_id)

    for i,t in enumerate(times):
        mesh[f"u_y_{i}"] = mesh[f"darcy_{i}"][:,1]

    u_y = evaluate_scalar(mesh, times, "u_y", pvs_id)

    pvs_axial_pore_pressure,y_coords = get_axial_mean(mesh, times, extra_p, domain_ids=pvs_id)
    #pvs_axial_total_pressure,_ = get_axial_mean(mesh, times, "phi_e", domain_ids=pvs_id)

    ef_axial_pore_pressure,_ = get_axial_mean(mesh, times, intra_p, domain_ids=astro_ids)
    #ef_axial_total_pressure,_ = get_axial_mean(mesh, times, "phi_i", domain_ids=astro_ids)

    pvs_axial_y_velocity,_ = get_axial_mean(mesh, times, "u_y", domain_ids=pvs_id)
    ef_axial_von_mises,_ = get_axial_mean(mesh, times, "vm", domain_ids=astro_ids)

    results = {
        "endfeet_von_mises":ef_vm,
        "pvs_von_mises":pvs_vm,
        "endfeet_pore_pressure":ef_p,
        "oc_pore_pressure":oc_p,
        "oc_von_mises":oc_vm,
        "pvs_pore_pressure":pvs_p,
        #"endfeet_total_pressure":ef_phi,
        #"pvs_total_pressure":pvs_phi,
        "mean_y_velocity": u_y,
        "pvs_axial_pore_pressure":pvs_axial_pore_pressure,
        #"pvs_axial_total_pressure":pvs_axial_total_pressure,
        "ef_axial_pore_pressure":ef_axial_pore_pressure,
        #"ef_axial_total_pressure":ef_axial_total_pressure,
        "pvs_axial_y_velocity":pvs_axial_y_velocity,
        "ef_axial_von_mises":ef_axial_von_mises,
        "axial_y_coordinates":y_coords,
        "times":times
    }

    if len(ecs_id) > 0:
        ecs_vm = evaluate_scalar(mesh, times, "vm", ecs_id)
        ecs_p = evaluate_scalar(mesh, times, extra_p, ecs_id)
        #ecs_phi = evaluate_scalar(mesh, times, "phi_e", ecs_id)
        results["ecs_von_mises"] = ecs_vm
        results["ecs_pore_pressure"] = ecs_p
        #results["ecs_total_pressure"] = ecs_phi

        ecs_axial_pore_pressure,_ = get_axial_mean(mesh, times, extra_p, domain_ids=ecs_id)
        #ecs_axial_total_pressure,_ = get_axial_mean(mesh, times, "phi_e", domain_ids=ecs_id)
        results["ecs_axial_pore_pressure"] = ecs_axial_pore_pressure
        #results["ecs_axial_total_pressure"] = ecs_axial_total_pressure

    for k,v in results.items():
        results[k] = np.array(v).tolist()

    with open(f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_pyvista.yml", 'w') as file:
        yaml.dump(results, file)
