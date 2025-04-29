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

with open(f"config_files/{sim_name}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

time_indices =  [config["num_steps"] -i for i in[22]]# [0, 8, 15, 22, 30]]
time_indices.sort()
print(time_indices)
cmap = Colormap("bids:inferno").to_matplotlib()

grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["darcy"], time_indices=time_indices)
num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]
ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]
cell_ids = oc_ids + ef_ids

ecs = grid.extract_cells(np.isin(grid["subdomains"], ecs_ids + pvs_ids))
ics = grid.extract_cells(np.isin(grid["subdomains"], cell_ids))

if cmax is None:
    absmax = max(minmax([(np.linalg.norm(grid[f"darcy_{ti}"], axis=1)) for ti in time_indices], percentile=95))
    clim = (0, + absmax)
    cmax = "perc99"
else: 
    clim = (0, cmax*1e-6)
    cmax = f"{cmax:.1f}"
    
folder = f"results/{mesh_name}_{sim_name}/flow_details_{cmax}"
os.makedirs(folder, exist_ok=True)

detail_pos= [(14.0, 1.7, 24) ,(14.0, 14.2, 5.3),]
n_streams = 5 #2
stream_width = 5
opacity = 1 #[0.3,0.8,1]
stream_args = dict(interpolator_type="cell", max_step_length=0.02,
 max_steps=1000, terminal_speed=0.2e-8, max_error=1e-8, min_step_length=0.005)

ics = ics.clip(normal="x", origin=(14.1,0,0))
es = ecs.slice(normal="x", origin=(14.1,0,0), generate_triangles=True)

for ti in time_indices:
    for si, pos in enumerate(detail_pos):
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))

        for cid, col in zip(ef_ids, Colormap("colorbrewer:brbg_6").iter_colors(6)):
            if cid in ics["subdomains"]:
                pl.add_mesh(ics.extract_cells(np.isin(ics["subdomains"],cid)),
                    color=col.rgba)
        pl.add_mesh(ics.extract_cells(np.isin(ics["subdomains"],oc_ids)), 
            color=occolor)

        stream = es.streamlines(source_center=pos, source_radius=2.5,n_points=18000,
            vectors=f"darcy_{ti}", surface_streamlines=True, **stream_args)

        stream[f"darcy_{ti}"][:,1] = 0
        pl.add_mesh(stream, scalars=f"darcy_{ti}", clim=clim,
            render_lines_as_tubes=True,
            cmap=cmap, line_width=stream_width, show_scalar_bar=False)

        pl.camera.focal_point = pos
        pl.camera.position = np.array(pos) + 10*np.array([1,0,0])
        pl.camera.elevation += 10
        pl.screenshot(folder + f"/detail_{si}_{ti}_{cmax}.png", transparent_background=True)


colfig, colax = create_colorbar(cmap, np.array(clim)*1e6, orientation="vertical", 
                                extend="max", figsize=(1.0, 3), label="flow velocity (μm/s)")
colfig.savefig(f"{folder}/colorbar_vertical_{cmax}.png", transparent=True, dpi=300)

colfig, colax = create_colorbar(cmap, np.array(clim)*1e6, orientation="horizontal", 
                                extend="max", figsize=(3, 1.0), right=0.9, bottom=0.6, label="flow velocity (μm/s)")
colfig.savefig(f"{folder}/colorbar_horizontal_{cmax}.png", transparent=True, dpi=300)

