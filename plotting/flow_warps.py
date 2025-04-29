from endfeet_buffering_visualisation_k3d import minmax, get_surface_interface_line
import pyvista as pv
import os
import numpy as np
from utils import pvscolor, ecscolor, efcolor, create_colorbar, read_fenics_results
import matplotlib.pyplot as plt
from cmap import Colormap
import pyvista as pv
import sys

def get_triangle(axis):
    return pv.Cone(radius=0.4).scale(0.2)
    tr = pv.Triangle([[-0.2,0, -0.07], [0, 0, 0], [-0.2, 0, 0.07]])
    if axis=="y": return tr
    if axis=="x": return tr.rotate_z(90)
    if axis=="z": return tr.rotate_x(90)


sim_name = "osmotic-pressure"
mesh_name = "artnew"

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

ax2num = {"x":0, "y":1, "z":2}

scale_fac = 20
time_indices =  [29] #[60, 68, 75, 82]
cmapinf = Colormap("bids:inferno").to_matplotlib()
cmap = Colormap("cmasher:prinsenvlag_r").to_matplotlib()

folder = f"results/{mesh_name}_{sim_name}/pressure_flow"
os.makedirs(folder, exist_ok=True)
grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["d", "darcy", "p"], time_indices=time_indices)
num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
domain_length = grid.bounds[3] - grid.bounds[2]
tol = 0.05*domain_length

ecs = grid.extract_cells(np.isin(grid["subdomains"], ecs_ids + pvs_ids))
ics = grid.extract_cells(np.isin(grid["subdomains"], cell_ids))
absmax = max(minmax([(np.linalg.norm(grid[f"darcy_{ti}"], axis=1)) for ti in time_indices], percentile=80))
clim = (0, absmax)
pmax = max(minmax([grid[f"p_{ti}"] for ti in time_indices], percentile=99))
plim = (-pmax, pmax)

colfig, colax = create_colorbar(cmapinf, np.array(clim)*1e6,
    orientation="vertical", figsize=(1.0, 2), label="flow velocity (μm/s)", extend="max")
colfig.savefig(f"{folder}/colorbar_flow.png", transparent=True, dpi=300)

colfig, colax = create_colorbar(cmap, plim,
    orientation="vertical", figsize=(1.0, 2), label="pore pressure (Pa)")
colfig.savefig(f"{folder}/colorbar_pressure.png", transparent=True, dpi=300)

for axis in ["x", "y", "z"]:
    ecs_slices = ecs.slice_along_axis(n=3, axis=axis, generate_triangles=True, tolerance=tol)
    if axis!="y": ecs_slices = [ecs_slices[1]]
    ics_slices = [ics.slice(normal=axis, origin=s.center, generate_triangles=True) for s in ecs_slices]
    complete_slice = [grid.slice(normal=axis, origin=s.center, generate_triangles=True) for s in ecs_slices]

    for i, (s1,s2,sc) in enumerate(zip(ecs_slices, ics_slices, complete_slice)):
        s = pv.merge([s1, s2], merge_points=False)
        mem = get_surface_interface_line(sc, pvs_ids + ecs_ids, cell_ids)

        for ti in time_indices:
            s[f"d_{ti}"][:, ax2num[axis]] = 0
            s.points += s[f"d_{ti}"] * 1e6 *scale_fac
            mem.points += mem[f"d_{ti}"] * 1e6 *scale_fac
            pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))
            pl.add_mesh(s, scalars=f"p_{ti}",cmap=cmap, clim=plim, show_scalar_bar=False)
            streams = s.streamlines_from_source(pv.PointSet(s.points[::20,:]),
                vectors=f"darcy_{ti}", surface_streamlines=True, 
                interpolator_type="cell", max_step_length=0.1, initial_step_length=0.1)
            
            streams[f"darcy_{ti}"][:, ax2num[axis]] = 0
            arrows = streams.glyph(scale=None, orient=f"darcy_{ti}", tolerance=0.03,
                geom=get_triangle(axis), factor=2.5)
            pl.add_mesh(streams, scalars=f"darcy_{ti}", clim=clim, cmap=cmapinf,
                line_width=2, show_scalar_bar=False, opacity=[0.2,0.8,1])
            pl.add_mesh(arrows, scalars=f"GlyphVector", clim=clim, cmap=cmapinf,
                show_scalar_bar=False, opacity=[0.4,1])
            pl.add_mesh(mem, color="gray", line_width=2)
            pl.camera_position = {"x":"yz", "y":"xz", "z":"xy"}[axis]
            pl.camera.zoom(1.3)
            pl.screenshot(folder + f"/{axis}_{i}_{ti}.png", transparent_background=True)

            s.points -= s[f"d_{ti}"] * 1e6 *scale_fac
            mem.points -= mem[f"d_{ti}"] * 1e6 *scale_fac
