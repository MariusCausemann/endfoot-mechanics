from endfeet_buffering_visualisation_k3d import read_results, minmax, get_surface_interface_line
import pyvista as pv
import os
import numpy as np
from utils import pvscolor, ecscolor, efcolor, create_colorbar
import matplotlib.pyplot as plt
from cmap import Colormap
import sys
import yaml


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    if len(sys.argv)>3:
        cmax = float(sys.argv[3])
    else: cmax = None

scale_fac = 20
cmap = Colormap("colorbrewer:PRGn").to_matplotlib()
with open(f"config_files/{sim_name}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

time_indices =  [config["num_steps"] -i for i in [1, 8, 15, 22, 30]]
time_indices.sort()
print(time_indices)

grid, bm, mesh_config, config = read_results(sim_name, mesh_name)
num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
domain_length = grid.bounds[3] - grid.bounds[2]
tol = 0.05*domain_length
times = np.linspace(0, config["T"], config["num_steps"])
A = config["vessel_pulsation"]
vw_disp = A * np.sin(2 * np.pi * config["f"] * times)

if cmax is None:
    absmax = max(minmax([grid[f"divd_{ti}"] for ti in time_indices], percentile=95), key=abs)
    clim = (- absmax, + absmax)
    cmax = "perc95"
else: 
    clim = (-cmax, cmax)
    cmax = f"{cmax:.1f}"

folder = f"results/{mesh_name}_{sim_name}/div_slices_{cmax}"
os.makedirs(folder, exist_ok=True)

for ti in time_indices:
    grid[f"divd_{ti}"] *= 100
    fig, ax = plt.subplots(figsize=(8,1.4))
    ax.plot(times, vw_disp, color="crimson", lw=6)
    ax.axis("off")
    ax.scatter(times[ti], vw_disp[ti], marker="*", color="black", s=1e3, zorder=3)
    ax.set_clip_on(False)
    ax.margins(y=0.3)
    fig.savefig(f"{folder}/vw_disp_cycle_{ti}.png", bbox_inches="tight", pad_inches=0.1)

slices = grid.slice_along_axis(n=3, axis="y", generate_triangles=False, tolerance=tol)

    
colfig, colax = create_colorbar(cmap, clim, orientation="vertical",
            figsize=(1.0, 3), label="volume change (%)")
colfig.savefig(f"{folder}/colorbar_{cmax}.png", transparent=True, dpi=300)
colfig, colax = create_colorbar(cmap, clim, orientation="horizontal",
            figsize=(2.5,1), right=0.9, bottom=0.7, label="volume change (%)")
colfig.savefig(f"{folder}/colorbar_horizontal_{cmax}.png", transparent=True, dpi=300)

for i,s in enumerate(slices):
    camera = None
    for ti in time_indices:
        s.points += s[f"d_{ti}"] * 1e6 *scale_fac
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))
        for sid in [pvs_ids, ecs_ids, cell_ids]:
            dom = s.extract_cells(np.isin(s["subdomains"], sid)).ctp()
            pl.add_mesh(dom, scalars=f"divd_{ti}",cmap=cmap, clim=clim, show_scalar_bar=False)
        mem = get_surface_interface_line(s, pvs_ids + ecs_ids, cell_ids)
        pl.add_mesh(mem, color="black", line_width=2, render_lines_as_tubes=True)
        if camera: pl.camera = camera
        else:
            pl.camera_position = 'xz'
            pl.camera.zoom(1.3)
            camera = pl.camera.copy()
        pl.screenshot(folder + f"/divd_{i}_{ti}_{cmax}.png", transparent_background=True)
        s.points -= s[f"d_{ti}"] * 1e6 *scale_fac
