
import os, sys
import meshio
import pyvista as pv
from biotstokes.IO import xdmf_to_unstructuredGrid

import numpy as np
import numpy_indexed as npi
import matplotlib.pyplot as plt
import yaml
from fenics import *

pv.start_xvfb()  
porous_id = 1
fluid_id = 2

def compute_lame_params(material_params):
    E = material_params["E"]
    nu = material_params["nu"]
    D_p = material_params["D_p"]
    material_params["mu_s"] = E/(2.0*(1.0+nu))
    material_params["lmbda"] = nu*E/((1.0-2.0*nu)*(1.0+nu))
    material_params["kappa"] = D_p*(1- 2*nu) / (2*E*(1 - nu))
    return material_params


def analyse_endfeet_results(name, mesh_name):

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    facet_ids = mesh_config["facet_ids"]

    with open(f"config_files/{name}.yml") as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

    material_params = config["material_parameters"]

    if config["type"] == "biotbiot":
        intra_cell_pressure = "pP_i"
        extra_cell_pressure = "pP_e"
        intra_mat_params = compute_lame_params(material_params["intracellular"])
        extra_mat_params = compute_lame_params(material_params["extracellular"])
        vessel_move = "d"
    if config["type"] == "biotstokes":
        intra_cell_pressure = "pP"
        extra_cell_pressure = "pF"
        intra_mat_params = compute_lame_params(material_params)
        extra_mat_params = compute_lame_params(material_params)
        vessel_move = "u"

    f = config["f"]
    T = config["T"]
    num_steps = config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps + 1)

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_comp_ids = sum(mesh_config["fluid_comp_ids"].values(), [])

    # +
    subd = pv.read(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    grid = xdmf_to_unstructuredGrid(f"results/{mesh_name}_{name}/{mesh_name}_{name}.xdmf", idx=[-10])
    print(grid.cell_data.keys())
    rounding = 12
    subd_c = subd.cell_centers()
    grid_c = grid.cell_centers()
    indices = npi.indices(np.round(subd_c.points, rounding), np.round(grid_c.points, rounding),
                        axis=0)

    subd.cell_data["subdomains"] = subd.cell_data["subdomains"][indices]
    grid.cell_data["subdomains"] = subd.cell_data["subdomains"]
    #from IPython import embed; embed()
    grid.points += grid["d"]*10

    fluid = grid.extract_cells(np.isin(grid.cell_data["subdomains"], fluid_comp_ids))
    endfeet = grid.extract_cells(np.isin(grid.cell_data["subdomains"], cell_ids))
    endfeet = endfeet.compute_derivative(scalars=intra_cell_pressure)
    endfeet["darcy"] =  - intra_mat_params["kappa"] / intra_mat_params["mu_f"] * endfeet["gradient"]

    bm = pv.read(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf")
    bm.point_data[vessel_move] = grid[vessel_move]
    vessel_wall = bm.extract_cells(np.isin(bm.cell_data["boundaries"], facet_ids["arterial_wall_id"] ) ).extract_surface()
    vessel_wall.compute_normals(inplace=True)

    # +
    #from IPython import embed; embed()
    pl = pv.Plotter(off_screen=True, window_size=[800, 800])
    pl.background_color="white"
    pl.add_mesh(endfeet, color="lightseagreen", opacity=1)
    #pl.add_mesh(outer_cells, color="cornflowerblue", opacity=0.5)

    #pl.add_arrows(cent=endfeet.points, direction=endfeet["darcy"], mag=1, color="yellow")
    #pl.add_arrows(cent=fluid.points, direction=fluid["u"], mag=1, color="white")
    #pl.add_arrows(cent=vessel_wall.points, direction=vessel_wall.point_arrays["Normals"], mag=1e-6, color="white")
    pl.add_mesh(vessel_wall, color="red", opacity=0.6)
    pl.camera.azimuth = -20
    pl.camera.elevation = - 20
    pl.screenshot(f"results/{mesh_name}_{name}/3D_animation.png")

    #from IPython import embed; embed()

    grid.clear_point_data()
    pv.save_meshio(f"results/{mesh_name}_{name}/subdomains_sorted.xdmf", grid)

    # +
    outer_cell_time_series = []
    endfeet_time_series = []
    ex_time_series = []
    vessel_time_series = []
    vessel_radius = np.linalg.norm(vessel_wall.points[:,[0,2]], axis=1).mean()
    p_max = -np.inf
    p_min = np.inf

    vessel_wall.clip(normal="x", inplace=True)

    print("load data...")
    for i in range(num_steps):
        print(i)
        grid = xdmf_to_unstructuredGrid(f"results/{mesh_name}_{name}/{mesh_name}_{name}.xdmf", idx=[i])
        grid.cell_data["subdomains"] = subd.cell_data["subdomains"]
        bounds = list(grid.bounds)
        bounds[0] = (bounds[0] + bounds[1]) * 0.5
        bounds[2] = (bounds[2] + bounds[3]) * 0.5

        grid = grid.clip_box(bounds)

        fluid = grid.extract_cells(np.isin(grid.cell_data["subdomains"], fluid_comp_ids ) )
        if config["type"]=="biotbiot":
            fluid = fluid.compute_derivative(scalars=extra_cell_pressure)
            fluid["u"] = - extra_mat_params["kappa"] / extra_mat_params["mu_f"] * fluid["gradient"]
        endfeet = grid.extract_cells(np.isin(grid.cell_data["subdomains"], cell_ids ) ).compute_derivative(scalars=intra_cell_pressure)
        endfeet["darcy"] =  - intra_mat_params["kappa"] / intra_mat_params["mu_f"] * endfeet["gradient"]
        endfeet.points += endfeet["d"]

        ex_time_series.append(fluid)
        endfeet_time_series.append(endfeet)
        vw = vessel_wall.copy().clip_box(bounds)
        v= -vw.point_arrays["Normals"]*np.cos(2*np.pi*i*dt*f)*config["vessel_pulsation"]
        vw.points += v
        vessel_time_series.append(vw)
        p_max = max(endfeet[intra_cell_pressure].max(), p_max)
        p_min = min(endfeet[intra_cell_pressure].min(), p_min)

    

    pv.set_jupyter_backend(None)
    sargspP = dict(height=0.5, vertical=True, position_x=0.05, position_y=0.05, color="black", 
                title_font_size=20, label_font_size=16, title="p [Pa]")
    sargspF = dict(sargspP)
    sargspF.update({"position_x":0.88})
    pl = pv.Plotter(window_size=[800, 800])
    pl.add_mesh(grid)
    pl.camera.azimuth = 10
    pl.camera.elevation = -20
    #pl.show()
    c = pl.camera
    pl.close()
    try:
        os.mkdir(f"results/{mesh_name}_{name}/animations_endfeet3D")
    except:
        pass

    print("create plots...")

    for i, endfeet, fluid, vw in zip(range(num_steps), endfeet_time_series, ex_time_series,
                                            vessel_time_series):
        pl = pv.Plotter(window_size=[800, 800],off_screen=True)
        pl.background_color="white"
        pl.add_mesh(endfeet, scalars=intra_cell_pressure, opacity=1, cmap="balance",
                    clim=(p_min*0.8, p_max*0.8), scalar_bar_args=sargspP)
        #pl.add_arrows(cent=endfeet.points, direction=endfeet["darcy"], mag=0.2, color="yellow")
        pl.add_arrows(cent=fluid.points, direction=fluid["u"], mag=0.04, color="white")
        pl.add_mesh(vw, color="red", opacity=0.6)
        pl.camera = c
        pl.screenshot(f"results/{mesh_name}_{name}/animations_endfeet3D/{name}_{i:03d}.png")
        pl.close()


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    analyse_endfeet_results(sim_name, mesh_name)

