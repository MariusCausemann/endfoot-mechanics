import pyvista as pv
import sys
from plotting.endfeet_buffering_visualisation_k3d import read_results
import numpy as np
import yaml
from multiprocessing import Pool
from functools import partial
import time
import os
from scipy.spatial import cKDTree as KDTree
from tqdm import tqdm

def map_kdtree(data_points, query_points, **kwargs):
    tree = KDTree(data_points)
    dist, idx = tree.query(query_points, **kwargs)
    return dist, idx

num_cpus = len(os.sched_getaffinity(0))

def compute_vessel_volume_changes(mesh, vw, num_steps):
    from tqdm import tqdm
    surf = mesh.extract_surface()
    surf.compute_normals(non_manifold_traversal=False, inplace=True)
    displacements = [surf[f"d_{i}"] for i in range(num_steps)]
    surf.clear_data()
    vol_vessel = [get_vessel_vol(vw.bounds, surf, d) for d in tqdm(displacements)]
    return  np.array([vol for vol,_ in vol_vessel]),  vol_vessel[0][1]

def get_vessel_vol(bounds, surf, displacement=0):
    b = np.array(bounds).reshape(3,2)
    b[1,0] += 0.1 
    b[1,1] -= 0.1
    xrng = np.linspace(*b[0], 100)
    yrng = np.linspace(*b[1], 100)
    zrng = np.linspace(*b[2], 100)
    grid = pv.RectilinearGrid(xrng, yrng, zrng)
    surf.points += displacement
    grid.compute_implicit_distance(surf, inplace=True)
    vessel = grid.threshold(0, invert=False).extract_largest()
    vessel.clear_data()
    surf.points -= displacement
    return vessel.volume, vessel

def evaluate_gap_sizes(mesh, domain_ids, object_ids, width_classes=None, displacement=0):
    mesh.points += displacement
    ecs = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], domain_ids ))
    surf = ecs.extract_surface()
    points = np.vstack([ecs.cell_centers().points, ecs.points])

    cc = pv.PointSet(points).compute_implicit_distance(surf)
    cell_midpoints = ecs.cell_centers().points
    d = abs(cc["implicit_distance"])
    ecs["gap_size"] = np.zeros(ecs.number_of_cells)
    if width_classes is None:
        width_classes = np.linspace(0, d.max(), 50, endpoint=False)
    for ri in np.array(width_classes) / 2:
        mask = d>=ri
        if mask.sum() > 0:
            dist, idx = map_kdtree(points[mask], cell_midpoints, distance_upper_bound=ri)
            ecs["gap_size"] = np.maximum(ecs["gap_size"],  2*ri*(dist<ri))

    ecs = ecs.compute_cell_sizes()
    return ecs.cell_data["gap_size"], abs(ecs.cell_data["Volume"])


def compute_gap_size_changes(mesh, mesh_config, num_steps, domain_ids, object_ids, width_classes):
    displacements = [mesh[f"d_{i}"] for i in range(num_steps)]
    empty_mesh = pv.UnstructuredGrid()
    empty_mesh.copy_structure(mesh)
    empty_mesh["subdomains"] = mesh["subdomains"]
    start = time.time()
    with Pool(num_cpus) as p:
        results = p.map(partial(evaluate_gap_sizes, empty_mesh,
                        domain_ids, object_ids, width_classes), displacements)
    distances = [r[0] for r in results]
    volumes = [r[1] for r in results]
    print(f"t: {time.time() - start:.2f} s")
    return distances, volumes


def gap_size_histo(distances, volumes, bins):  
    counts = []
    for d,v in zip(distances, volumes):
        c, bin_edges = np.histogram(d, weights=v, bins=bins, density=True)
        counts.append(c)
    midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
    return counts, midpoints

def extract_line_data(mesh, num_steps, a, b, resolution, scalars="subdomains"):
    points = np.array(mesh.points)
    line_data = []
    for i in range(num_steps):
        mesh.points = points + mesh[f"d_{i}"]
        data = mesh.sample_over_line(a, b, resolution=resolution)[scalars]
        line_data.append(data)
    return line_data


def compute_cross_section_area(mesh, domain_ids=None, n=3, axis="y"):
    if domain_ids:
        mesh = mesh.extract_cells(np.isin(mesh["subdomains"], domain_ids))
    length = mesh.bounds[3] - mesh.bounds[2]
    slices = mesh.slice_along_axis(n=n, axis=axis, tolerance=0.05*length)
    areas = [sl.area for sl in slices]
    coords = [sl.bounds[2] for sl in slices]
    return areas, coords

if __name__ == "__main__":
    start = time.time()
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    mesh, bm, mesh_config, config = read_results(sim_name, mesh_name)
    astro_ids = mesh_config["porous_cell_ids"]["astrocyte"]
    object_ids = astro_ids
    object_ids += mesh_config["porous_cell_ids"].get("outer_cell", [])
    ecs_id = mesh_config["fluid_comp_ids"]["ecs"]
    pvs_id = mesh_config["fluid_comp_ids"]["pvs"]

    num_steps = config["num_steps"]
    dt = config["T"] / num_steps
    times = np.linspace(0, num_steps*dt, num_steps)
    results = dict()
    results["times"] = times
    print(f"read data: {time.time() - start:.2f} s")
    start = time.time()
    for i in range(num_steps):
        mesh[f"d_{i}"] *= 1e6

    vw = bm.extract_cells(bm["boundaries"]==mesh_config["facet_ids"]["arterial_wall_id"])
    vol, vessel = compute_vessel_volume_changes(mesh, vw, 1)
    #from IPython import embed; embed()
    vessel_area, y_coords = compute_cross_section_area(vessel, n=100)
    pvs_area, y_coords = compute_cross_section_area(mesh, domain_ids=[pvs_id], n=100)
    ef_area, y_coords = compute_cross_section_area(mesh, domain_ids=astro_ids, n=100)

    results["y_coordinates"] = y_coords
    results["pvs_area"] = pvs_area
    results["ef_area"] = ef_area
    results["vessel_area"] = vessel_area

    print(f"compute cross section areas: {time.time() - start:.2f} s")
    start = time.time()
    
    length = mesh.bounds[3] - mesh.bounds[2]
    vessel_vol, _ = compute_vessel_volume_changes(mesh, vw, num_steps)
    vessel_diam = np.sqrt(vessel_vol / (length *np.pi)) * 2

    results["vessel_diam"] = vessel_diam

    results["vessel_volume"] = vessel_vol

    start = time.time()
    if len(ecs_id) > 0:
        ecs_width_classes = np.linspace(0, 1, 100, endpoint=False)
        distances, ecs_volumes = compute_gap_size_changes(mesh, mesh_config,
                                num_steps, ecs_id, object_ids, ecs_width_classes)
        print(f"compute ecs gap sizes: {time.time() - start:.2f} s")
        start = time.time()
        ecs_counts, ecs_midpoints = gap_size_histo(distances, ecs_volumes, ecs_width_classes)
        print(f"compute ecs gap size histo: {time.time() - start:.2f} s")
        start = time.time()
        ef_with_classes = np.linspace(0.025, 0.075, 50)
        distances, ecs_volumes = compute_gap_size_changes(mesh, mesh_config,
                                num_steps, ecs_id, object_ids, ef_with_classes)
        ef_counts, ef_midpoints = gap_size_histo(distances, ecs_volumes, ef_with_classes)
        results["ecs_volume_counts"] = ecs_counts
        results["ecs_volume_binpoints"] = ecs_midpoints
        results["ef_volume_counts"] = ef_counts
        results["ef_volume_binpoints"] = ef_midpoints

    pvs_id = mesh_config["fluid_comp_ids"]["pvs"]
    pvs_with_classes = np.linspace(0, 3, 100)
    distances, pvs_volumes = compute_gap_size_changes(mesh, mesh_config,
                            num_steps, pvs_id, object_ids, pvs_with_classes)
    pvs_counts, pvs_midpoints = gap_size_histo(distances, pvs_volumes, pvs_with_classes)

    results["pvs_volume_counts"] = pvs_counts
    results["pvs_volume_binpoints"] = pvs_midpoints
    results["sim_config"] = config
    results["mesh_config"] = mesh_config

    for k,v in results.items():
        results[k] = np.array(v).tolist()
    with open(f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_ts.yml", 'w') as file:
        yaml.dump(results, file)
    