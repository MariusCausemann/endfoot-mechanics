from endfeet_buffering_visualisation_k3d import minmax, get_surface_interface_line
import pyvista as pv
import os
import numpy as np
from utils import pvscolor, ecscolor, efcolor, occolor, create_colorbar, read_fenics_results
import matplotlib.pyplot as plt
from cmap import Colormap
import pyvista as pv
import sys
import yaml

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    if len(sys.argv)>3:
        cmax = float(sys.argv[3])
    else: cmax = None

plim = (-cmax,cmax)
arrow_factor = 0.75e6 if "open" in sim_name else 1.5e7
with open(f"config_files/{sim_name}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

time_indices =  [config["num_steps"] -i for i in [22, ]]#8, 15, 22, 30]]
time_indices.sort()
print(time_indices)
cmap = Colormap("cmasher:prinsenvlag_r").to_matplotlib()
folder = f"results/{mesh_name}_{sim_name}/flow_clip"
os.makedirs(folder, exist_ok=True)

grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["darcy", "p"], time_indices=time_indices)

pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]
ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]
cell_ids = oc_ids + ef_ids

origin = (14.0,0,0)
ecs = grid.extract_cells(np.isin(grid["subdomains"], ecs_ids + pvs_ids))
ics = grid.extract_cells(np.isin(grid["subdomains"], cell_ids))
clip = pv.merge([ecs.clip(normal="x", origin=origin), ics.clip(normal="x",origin=origin)], merge_points=False)
pvs = grid.extract_cells(np.isin(grid["subdomains"], pvs_ids))

p = pv.Plane(center=grid.center, direction=(1,0,0), i_size=16,j_size=18,
     i_resolution=14,j_resolution=10).sample(pvs, tolerance=2)
scale_mag = 1.5
for ti in time_indices:
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))
    pl.add_mesh(clip, scalars=f"p_{ti}", show_scalar_bar=False, clim=plim,cmap=cmap)
    p[f"darcy_{ti}"][:,0] = 0
    arrows = p.glyph(orient=f"darcy_{ti}", scale=f"darcy_{ti}", factor=arrow_factor)
    pl.add_mesh(arrows, color="black")
    pl.add_arrows(cent=np.array([14, 10, 26]), direction=np.array([0, -1, 0]), 
                  mag=scale_mag, show_scalar_bar=False, color="black")
    pl.camera_position = "yz"
    pl.camera.zoom(1.3)
    pl.screenshot(folder + f"/clip_{ti}.png", transparent_background=True)
    #pl.export_html(folder + f"/clip_{ti}.html")

colfig, colax = create_colorbar(cmap, plim, orientation="vertical", 
                                extend=None, figsize=(1.0, 3), label="pressure (Pa)")
colfig.savefig(f"{folder}/colorbar_vertical_{cmax}.png", transparent=True, dpi=300)

colfig, colax = create_colorbar(cmap, plim, orientation="horizontal", 
                                extend=None, figsize=(3, 1.0), right=0.9, bottom=0.6, label="pressure (Pa)")
colfig.savefig(f"{folder}/colorbar_horizontal_{cmax}.png", transparent=True, dpi=300)

print(f"arrow scale: {scale_mag* 1e6 / arrow_factor}")
