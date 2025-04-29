import os, sys
import pyvista as pv
try:
    from .io_utils import xdmf_to_unstructuredGrid
except:
        from io_utils import xdmf_to_unstructuredGrid
import itertools
import numpy as np
import numpy_indexed as npi
import matplotlib as mpl
import matplotlib.pyplot as plt
import yaml
import k3d

hexcolor = lambda c: int(mpl.colors.to_hex(c)[1:], base=16)

slow_down = 4
time = lambda t: str(np.round(t / slow_down, 3))

camera = [3.5e-05, -4e-05, 3e-05,
          2e-05, 10e-06, 1.2e-05,
          0.0, 0.0, 1.0]
camera_end = [5e-05, -4e-05, 3e-05,
          2e-05, 10e-06, 1.2e-05,
          0.0, 0.0, 1.0]
camera = np.array(camera)*1e6
camera_end = np.array(camera_end)*1e6
camera = None
#camera = np.linspace(camera, camera_end, 91)

def minmax(arr_list, percentile=95):
    if percentile is None:
        return k3d.helpers.minmax(arr_list)
    else:
        percentiles = [np.percentile(arr, [100 - percentile, percentile])
                       for arr in arr_list]
        return k3d.helpers.minmax(percentiles)

def from_k3d(colorlist):
    cm = np.array(colorlist).reshape(-1, 4)[:, 1:]
    return mpl.colors.LinearSegmentedColormap.from_list("", cm)


def get_surf(mesh):
    mesh = mesh.extract_surface()
    mesh.compute_normals(inplace=True, non_manifold_traversal=False, consistent_normals=True)
    return mesh

def create_colorbar(cmap, clim, figsize=(0.2, 3), right=0.3, bottom=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=ax, **kwargs)
    fig.subplots_adjust(right=right, bottom=bottom)
    return fig, ax



def read_results(sim_name, mesh_name, time_indices=None):
    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    with open(f"config_files/{sim_name}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

    f = config["f"]
    T = config["T"]
    num_steps = config["num_steps"]
    if not time_indices:
        time_indices = range(num_steps)

    grid = xdmf_to_unstructuredGrid(
        f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf", idx=time_indices
    )

    # match subdomain marker to results (might have changed due to parallel execution)
    subd = pv.read_meshio(f"meshes/{mesh_name}/{mesh_name}.xdmf")

    indices = subd.find_closest_cell(grid.cell_centers().points)
    subd.cell_data["subdomains"] = subd.cell_data["subdomains"][indices]
    grid.cell_data["subdomains"] = subd.cell_data["subdomains"]
    bm = pv.read_meshio(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf")

    assert np.isclose(grid.points, bm.points).all()
    for t in time_indices:
        bm.point_data[f"d_{t}"] = grid.point_data[f"d_{t}"]
        bm.point_data[f"d_{t}"][:, 1] = 0

    #for t in time_indices[1:]:
    #    grid[f"darcy_{t}"] += (grid[f"d_{t}"] - grid[f"d_{t - 1}"]) / (T/num_steps)

    grid.cell_data["subdomains"] = subd.cell_data["subdomains"]

    if "pP_i_0" not in grid.array_names:

        ics_ids = sum(mesh_config["porous_cell_ids"].values(), [])
        ecs_ids = sum(mesh_config["fluid_comp_ids"].values(), [])
        grid["ids"] = range(grid.n_points)

        ecs_cell_mask = np.isin(grid.cell_data["subdomains"], ecs_ids)
        ics_cell_mask = np.isin(grid.cell_data["subdomains"], ics_ids)

        ecs = grid.extract_cells(ecs_cell_mask).cell_data_to_point_data()
        ics = grid.extract_cells(ics_cell_mask).cell_data_to_point_data()


        ecs_point_mask = np.isin(grid["ids"], ecs["ids"])
        ics_point_mask = np.isin(grid["ids"], ics["ids"])

        for i in time_indices:
            grid[f"pP_e_{i}"] = np.zeros(grid.n_points)
            grid[f"pP_i_{i}"] = np.zeros(grid.n_points)
            grid[f"pP_e_{i}"][ecs_point_mask] = ecs[f"p_{i}"]
            grid[f"pP_i_{i}"][ics_point_mask] = ics[f"p_{i}"]

    bm.scale(1e6, inplace=True)
    grid.scale(1e6, inplace=True)
    return grid, bm, mesh_config, config

def generate_boundary_visualization(name, mesh_name):
    grid, bm, mesh_config, config = read_results(name, mesh_name)
    facet_ids = mesh_config["facet_ids"]
    boundaries = [
        ("Vessel wall" , facet_ids["arterial_wall_id"]),
        ("PVS back" , facet_ids["pvs_inlet_id"]),
        ("PVS front" , facet_ids["pvs_outlet_id"]),
        ("ECS axial" , facet_ids["ecs_axial_id"]),
        ("ECS radial" , facet_ids["ecs_outer_id"]),
        ("EF radial" , facet_ids["cells_outer_id"]),
        ("EF axial" , facet_ids["cells_axial_id"]),
    ]

    colors = {
        "Vessel wall":hexcolor("firebrick"),
        "PVS front":hexcolor("midnightblue"),
        "PVS back":hexcolor("lightsteelblue"),
        "ECS axial":hexcolor("mediumpurple"),
        "ECS radial":hexcolor("indigo"),
        "EF radial":hexcolor("limegreen"),
        "EF axial":hexcolor("darkgreen"),
        }
    
    k3d_objs = []
    texts = []
    for i, (b_name, bid) in enumerate(boundaries):
        b = bm.extract_cells(
            np.isin(bm.cell_data["boundaries"], bid)
        ).extract_surface()
        b.compute_normals(inplace=True, non_manifold_traversal=False)
        c = colors[b_name]
        b_k3d = k3d.vtk_poly_data(b, side="double", color=c, name=b_name)
        k3d_objs.append(b_k3d)
        text = k3d.text2d(b_name,
                       position=(0.05, 0.2 + i*0.08 ), reference_point='lb',
                       color=c, size=3, is_html=True, label_box=False)
        texts.append(text)
    filename = f"results/{mesh_name}_{name}/boundaries/boundaries"
    generate_screenshots(k3d_objs + texts, 30, filename + "_white", background_color=hexcolor("white"),
                         camera=camera)
    generate_screenshots(k3d_objs + texts, 30, filename, camera=camera)


def generate_interface_visualization(name, mesh_name):
    grid, bm, mesh_config, config = read_results(name, mesh_name)
    facet_ids = mesh_config["facet_ids"]

    boundaries = [
        ("low AQP4" , facet_ids["interf_id"]),
        ("high AQP4" , facet_ids["aqp_membrane_id"]),
        ("EF radial" , facet_ids["cells_outer_id"]),
        ("EF axial" , facet_ids["cells_axial_id"]),
    ]
    colors = {
        "high AQP4":hexcolor("aquamarine"),
        "low AQP4":hexcolor("salmon"),
        "EF radial":hexcolor("limegreen"),
        "EF axial":hexcolor("darkgreen"),

        }
    
    bounds = list(bm.bounds)
    bounds[0] = (bounds[0] + bounds[1]) * 0.5
    bounds[3] = (bounds[2] + bounds[3]) * 0.5
    bm = bm.clip_box(bounds)
        
    k3d_objs = []
    texts = []
    for i, (b_name, bid) in enumerate(boundaries):
        b = bm.extract_cells(
            np.isin(bm.cell_data["boundaries"], bid)
        ).extract_surface()
        b.compute_normals(inplace=True, non_manifold_traversal=False)
        c = colors[b_name]
        b_k3d = k3d.vtk_poly_data(b, side="double", color=c, name=b_name)
        k3d_objs.append(b_k3d)
        text = k3d.text2d(b_name,
                       position=(0.05, 0.2 + i*0.08 ), reference_point='lb',
                       color=c, size=3, is_html=True, label_box=False)
        texts.append(text)

    filename = f"results/{mesh_name}_{name}/interface/interface"
    generate_screenshots(k3d_objs + texts, 30, filename + "_white", background_color=hexcolor("white"),
                         camera=camera)
    generate_screenshots(k3d_objs + texts, 30, filename, camera=camera)

def generate_interactive_html(name, mesh_name, tracer=True, 
    fluid_arrows=False, cross_section_slices=False,
    plotname="k3d_main"):

    grid, bm, mesh_config, config = read_results(name, mesh_name)
    visualization_T = 10
    vel_scale = 0.2 if "open" in name else 1
    disp_scale = 1e6*10
    T = config["T"]
    html_slow_down = visualization_T / T
    times = np.linspace(0, config["T"], config["num_steps"])
    html_t = [str(t*html_slow_down) for t in times]
    k3d_objs = []
    time_indices = range(config["num_steps"])

    intra_cell_pressure = "pP_i"
    extra_cell_pressure = "pP_e"
    facet_ids = mesh_config["facet_ids"]
    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_comp_ids = sum(mesh_config["fluid_comp_ids"].values(), [])

    bounds = list(grid.bounds)
    bounds[0] = (bounds[0] + bounds[1]) * 0.5
    bounds[3] = (bounds[2] + bounds[3]) * 0.5
    grid = grid.clip_box(bounds)
    if tracer:
        tracer = particle_tracer(grid, bm, mesh_config, config,
                                num_points=10000,accelerate_fac=1)
        k3d_objs += tracer

    vessel_wall = bm.extract_cells(
        np.isin(bm.cell_data["boundaries"], facet_ids["arterial_wall_id"])
    )
    vessel_wall = vessel_wall.clip_box(bounds).extract_surface()
    vessel_wall.compute_normals(inplace=True, non_manifold_traversal=False)

    fluid = grid.extract_cells(np.isin(grid.cell_data["subdomains"], fluid_comp_ids))
    endfeet = grid.extract_cells(np.isin(grid.cell_data["subdomains"], cell_ids))
    domain_size = abs(fluid.bounds[2] - fluid.bounds[3])

    endfeet_surf = get_surf(endfeet)
    color_map = k3d.paraview_color_maps.Erdc_iceFire_H
    endfeet_k3d = k3d.vtk_poly_data(
        endfeet_surf,
        color_attribute=(f"{intra_cell_pressure}_0", -1, 1),
        color_map=color_map,
        side="double",
        name="endfeet",
    )

    endfeet_k3d.vertices = {
        html_t[t]: endfeet_k3d.vertices + disp_scale * endfeet_surf[f"d_{t}"]
        for t in time_indices
    }
    endfeet_k3d.attribute = {
        html_t[t]: endfeet_surf[f"{intra_cell_pressure}_{t}"] for t in time_indices
    }
    domain_length = fluid.bounds[3] - fluid.bounds[2]
    fluid_slice = pv.merge(fluid.slice_along_axis(n=5, axis="y", tolerance=domain_length*0.05)).compute_normals(
        inplace=True, non_manifold_traversal=False
    )
    k3d_objs += endfeet_k3d
    if fluid_arrows:
        arrow_dom = fluid_slice.extract_cells(
            fluid_slice["subdomains"] == mesh_config["fluid_comp_ids"]["pvs"]
        )

        arr_max = max(
            [np.linalg.norm(arrow_dom[f"darcy_{t}"], axis=1).max() for t in time_indices]
        )
        arrows = [
            arrow_dom.glyph(
                scale=f"darcy_{t}",
                orient=f"darcy_{t}",
                tolerance=0.01,
                factor=1 / arr_max * domain_size * vel_scale,
            )
            for t in time_indices
        ]

        arrows_k3d = k3d.vtk_poly_data(arrows[3], name="flow arrows", color=hexcolor("purple"))
        arrows_k3d.vertices = {
            html_t[t]: arrows[t].points + disp_scale * arrows[t][f"d_{t}"]
            for t in time_indices
        }
        k3d_objs += arrows_k3d

    min_max = minmax(list(endfeet_k3d.attribute.values()))
    if cross_section_slices:
        fluid_slice_k3d = k3d.vtk_poly_data(
            fluid_slice,
            side="double",
            color_attribute=(f"{extra_cell_pressure}_0", -1, 1),
            color_map=color_map,
            name="slices",
            opacity=0.6,
        )
        # only move slice in radial direction
        for t in time_indices:
            fluid_slice[f"d_{t}"][:, 1] = 0
        fluid_slice_k3d.vertices = {
            html_t[t]: fluid_slice_k3d.vertices + disp_scale * fluid_slice[f"d_{t}"]
            for t in time_indices
        }
        fluid_slice_k3d.attribute = {
            html_t[t]: fluid_slice[f"{extra_cell_pressure}_{t}"] for t in time_indices
        }
        min_max2 = minmax(list(fluid_slice_k3d.attribute.values()))
        min_max = minmax([min_max, min_max2])
        fluid_slice_k3d.color_range = min_max
        k3d_objs += fluid_slice_k3d

    endfeet_k3d.color_range = min_max

    vessel_wall_k3d = k3d.vtk_poly_data(
        vessel_wall, side="double", color=0xFF6A6A, name="vessel wall", opacity=0.6
    )
    vessel_wall_k3d.vertices = {
        html_t[t]: vessel_wall_k3d.vertices + disp_scale * vessel_wall[f"d_{t}"]
        for t in time_indices
    }
    k3d_objs += vessel_wall_k3d

    text = k3d.text2d("Pa",
            position=(0.09, 0.83 ), reference_point='lb',
            color=hexcolor("white"), size=1.2, is_html=True, label_box=False)
    k3d_objs += text

    filename = f"results/{mesh_name}_{name}/{plotname}/{plotname}"
    frames = generate_screenshots(k3d_objs, 30, filename, times=times,
                                 html_slow_down=html_slow_down, camera=camera)
    filename_w = f"results/{mesh_name}_{sim_name}/{plot_name}_white/{plot_name}_white"

    frames = generate_screenshots(k3d_objs, 30, filename_w, times=times, background_color=hexcolor("white"),
                                 html_slow_down=html_slow_down, camera=camera)
    with open(f"{filename}.yml","w") as f:
        yaml.dump({"scalar_range":min_max, "cbar_label":"pressure (Pa)", "cmap":color_map}, f)
    plt.style.use("dark_background")
    fig, ax = create_colorbar(from_k3d(color_map), min_max, orientation="horizontal", 
        figsize=(3, 1.0), right=0.9, bottom=0.6, label="pressure (Pa)")
    fig.savefig(f"results/{mesh_name}_{sim_name}/{plot_name}/colorbar.png", dpi=150)


def get_surface_interface_line(grid, ids1, ids2):

    surf = grid.extract_geometry()
    surf.point_data["ids"] = range(surf.number_of_points)

    s1 = surf.extract_cells(np.isin(surf["subdomains"], ids1))
    s2 = surf.extract_cells(np.isin(surf["subdomains"], ids2))
    wf1 = s1.extract_all_edges()
    wf2 = s2.extract_all_edges()
    cell_idx = npi.indices(wf1.cell_centers().points, wf2.cell_centers().points, axis=0, missing="mask")
    interf = wf1.extract_cells(np.unique(cell_idx))
    return interf


def create_scalar_plot(grid, config, scalar_ids, scalars, cmaps,
                       vectors=(), vector_ids=(),cbar_label=None, plot_name=None,
                       visualization_T=10, disp_scale=1e7,
                       vec_color="white", vec_string=""):
    T = config["T"]
    html_slow_down = visualization_T / T
    times = np.linspace(0, config["T"], config["num_steps"])
    html_t = [str(t*html_slow_down) for t in times]
    bounds = list(grid.bounds)
    bounds[0] = (bounds[0] + bounds[1]) * 0.5
    bounds[3] = (bounds[2] + bounds[3]) * 0.5
    grid = grid.clip_box(bounds)
    domain_size = abs(grid.bounds[2] - grid.bounds[3])
    time_indices = range(config["num_steps"])
    k3d_objs = []
    for ids, scalar in zip(scalar_ids, scalars):
        subd = get_surf(grid.extract_cells(np.isin(grid.cell_data["subdomains"],ids)))
        if f"{scalar}_0" in subd.point_data.keys():
            subd_k3d = k3d.vtk_poly_data(subd,
                color_attribute=(f"{scalar}_0", -1, 1), side="double"
                )
        else:
            subd_k3d = k3d.vtk_poly_data(subd,
            cell_color_attribute=(f"{scalar}_0", -1, 1), side="double"
            )

        subd_k3d.attribute = {html_t[t]: subd[f"{scalar}_{t}"] for t in time_indices}
        subd_k3d.vertices = {
            html_t[t]: subd_k3d.vertices + disp_scale * subd[f"d_{t}"]for t in time_indices}
        k3d_objs.append(subd_k3d)

    for obj_ids, cmap, crange in cmaps:
        min_max = minmax(list(k3d_objs[obj_ids[0]].attribute.values()))
        for i in obj_ids[1:]:
            min_max2 = minmax(list(k3d_objs[i].attribute.values()))
            min_max = minmax([min_max, min_max2])
        for i in obj_ids:
            k3d_objs[i].color_map = cmap
            if not crange:
                crange = min_max
            k3d_objs[i].color_range = crange

    if len(scalar_ids)==2:
        edges = get_surface_interface_line(grid, scalar_ids[0], scalar_ids[1]).extract_geometry()
        l = np.delete(edges.lines, np.arange(0, edges.lines.size, 3))
        contour = k3d.lines(
            edges.points, l, indices_type="segment", width=0.2, color=0x66CD00
        )
        contour.vertices = {
            html_t[t]: contour.vertices + disp_scale * edges[f"d_{t}"]for t in time_indices}
        k3d_objs.append(contour)

    for ids, vec in zip(vector_ids, vectors):
        subd = get_surf(grid.extract_cells(np.isin(grid.cell_data["subdomains"],ids)))
        arr_max = max([np.linalg.norm(subd[f"{vec}_{t}"], axis=1).max()
            for t in time_indices])
        arrows = [
            subd.glyph(
                scale=f"{vec}_{t}",
                orient=f"{vec}_{t}",
                tolerance=0.03,
                factor= 1 / arr_max * domain_size * 0.1,
            )
            for t in time_indices
        ]
    
        arrows_k3d = k3d.vtk_poly_data(arrows[1], color=hexcolor(vec_color))
        arrows_k3d.vertices = {html_t[t]: arrows[t].points for t in time_indices}
        k3d_objs.append(arrows_k3d) 

    text = k3d.text2d(cbar_label,
                position=(0.09, 0.83 ), reference_point='lb',
                color=hexcolor("white"), size=1.2, is_html=True, label_box=False)

    #k3d_objs.append(text)
    filename = f"results/{mesh_name}_{sim_name}/{plot_name}/{plot_name}"
    frames = generate_screenshots(k3d_objs, 30, filename, times=times,
                                 html_slow_down=html_slow_down, camera=camera)
    filename_w = f"results/{mesh_name}_{sim_name}/{plot_name}_white/{plot_name}_white"
    frames = generate_screenshots(k3d_objs, 30, filename_w, times=times,
                                  background_color=hexcolor("white"),
                                  html_slow_down=html_slow_down, camera=camera)
    
    with open(f"{filename}.yml","w") as f:
        yaml.dump({"scalar_range":crange, "cbar_label":cbar_label, "cmap":cmap}, f)
    
    plt.style.use("dark_background")
    fig, ax = create_colorbar(from_k3d(cmap), crange, orientation="horizontal", 
        figsize=(3, 1.0), right=0.9, bottom=0.6, label=cbar_label)
    fig.savefig(f"results/{mesh_name}_{sim_name}/{plot_name}/colorbar.png", dpi=150)

def particle_tracer(grid, bm, mesh_config, config, num_points=10000,accelerate_fac=1):
    T = config["T"]
    nt = config["num_steps"]
    visualization_T = 10
    html_slow_down = visualization_T / T
    times = np.linspace(0, T, nt)
    html_t = [str(t*html_slow_down) for t in times]

    dt = T / nt
    facet_ids = mesh_config["facet_ids"]
    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_comp_ids = sum(mesh_config["fluid_comp_ids"].values(), [])
    pvs = grid.extract_cells(np.isin(grid.cell_data["subdomains"],mesh_config["fluid_comp_ids"]["pvs"]))
    bounds = np.array(pvs.bounds).reshape(3,2)
    points = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(num_points,3))
    inside = abs(pvs.probe(points)["pP_e_10"]) > 0
    points = points[inside]
    locations = np.zeros((nt + 1, *points.shape))
    locations[0, :,:] = points
    for i in range(config["num_steps"]):
        locations[i + 1,:,:] = (locations[i,:,:] 
        + 1e6*dt*grid.probe(locations[i,:,:])[f"darcy_{i}"]*accelerate_fac)
    moved = (locations[-1,:,:] - locations[-2,:,:]).sum(axis=1) !=0
    locations = locations[:,moved,:]
    points = k3d.points(locations[0,:,:] ,point_size=0.3,color=0x3e3a3a)
    points.positions = {html_t[t]:locations[t,:,:] for t in range(nt)}
    return points


if __name__ == "__main__":
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    plot_name = sys.argv[3]
    from headless import generate_screenshots
    #test_hardware_acceleration()
    grid, bm, mesh_config, config = read_results(sim_name, mesh_name)
    T = config["T"]
    facet_ids = mesh_config["facet_ids"]
    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_comp_ids = sum(mesh_config["fluid_comp_ids"].values(), [])
    color_map = k3d.paraview_color_maps.Erdc_iceFire_H
    if "osmotic" in sim_name:
        color_map = k3d.paraview_color_maps.Cool_to_Warm

    if plot_name=="pore_pressure":
        scalars = ["pP_i", "pP_e"]
        ids = [cell_ids, fluid_comp_ids]
        cmaps = [([0, 1], color_map, (-1.5, 1.5))]
        create_scalar_plot(grid, config,  ids, scalars,
         cmaps, plot_name=plot_name, cbar_label="pressure (Pa)")

    if plot_name=="displacement":
        for i in range(config["num_steps"]):
            grid[f"d_abs_{i}"] = np.linalg.norm(grid[f"d_{i}"], axis=1)*1e6
        scalars = ["d_abs", "d_abs"]
        ids = [cell_ids, fluid_comp_ids]
        color_map = k3d.paraview_color_maps.Yellow_15
        cmaps = [([0, 1], color_map, None )]
        vec_ids = [fluid_comp_ids, cell_ids]
        vectors = ["d","d"]
        create_scalar_plot(grid, config, ids, scalars, cmaps,
                         vector_ids=vec_ids, vectors=vectors,
                         plot_name=plot_name, cbar_label="displacement (μm)",
                         vec_color="aqua") # mu \u03BC
    
    if plot_name=="vm":
        scalars = ["vm", "vm"]
        grid2 = grid.cell_data_to_point_data()
        grid2["subdomains"] = grid["subdomains"]
        ids = [cell_ids, fluid_comp_ids]
        color_map = k3d.paraview_color_maps.Yellow_15
        cmaps = [([0, 1], color_map, None)]
        create_scalar_plot(grid2, config, ids, scalars, cmaps,
                         plot_name=plot_name, cbar_label="stress (Pa)")

    if plot_name=="interface":
        generate_interface_visualization(sim_name, mesh_name)

    if plot_name=="boundaries":
        generate_boundary_visualization(sim_name, mesh_name)

    if plot_name=="k3d_main":
        generate_interactive_html(sim_name, mesh_name)

    if plot_name=="k3d_main_arrows":
        generate_interactive_html(sim_name, mesh_name, fluid_arrows=True,
        tracer=False, cross_section_slices=False, plotname="k3d_main_arrows")

    if plot_name=="k3d_main_arrows_slices":
        generate_interactive_html(sim_name, mesh_name, fluid_arrows=True,
        tracer=False, cross_section_slices=True,plotname="k3d_main_arrows_slices")




