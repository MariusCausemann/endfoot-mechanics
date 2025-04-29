from endfeet_buffering_visualisation_k3d import minmax, get_surface_interface_line
import pyvista as pv
import os
import numpy as np
from utils import pvscolor, ecscolor, efcolor, occolor, create_colorbar, read_fenics_results
import matplotlib.pyplot as plt
from cmap import Colormap
import sys
import yaml

def get_triangle(axis):
    return pv.Triangle([[-0.2,0, -0.07], [0, 0, 0], [-0.2, 0, 0.07]])

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

time_indices =  [60, 68, 75, 82]
cmapinf = Colormap("bids:inferno").to_matplotlib()

cmap = Colormap("cmasher:prinsenvlag_r").to_matplotlib()
folder = f"results/{mesh_name}_{sim_name}/pressure_stack"
os.makedirs(folder, exist_ok=True)
print("reading results...")

grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["darcy", "p"], time_indices=time_indices)

num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]
ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]
cell_ids = oc_ids + ef_ids
domain_length = grid.bounds[3] - grid.bounds[2]
tol = 0.04*domain_length
times = np.linspace(0, config["T"], config["num_steps"])

slices = grid.slice_along_axis(n=3, axis="y", tolerance=tol)
facets = pv.read(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf").scale(1e6)
vw = facets.extract_cells(facets["boundaries"]==mesh_config["facet_ids"]["arterial_wall_id"])
pvs_mem = facets.extract_cells(facets["boundaries"]==mesh_config["facet_ids"]["aqp_membrane_id"])

for ti in time_indices:
    absmax = max(minmax(slices.combine()[f"p_{ti}"], percentile=None))
    clim = (-absmax, absmax)
    pl = pv.Plotter(off_screen=True, window_size=(2400, 2000))
    for s in slices:
        pl.add_mesh(s, scalars=f"p_{ti}", clim=clim, cmap=cmap, show_scalar_bar=False)
        mem_slice = pvs_mem.slice(normal="y", origin=s.center)
        pl.add_mesh(mem_slice, color="green",
            line_width=6, render_lines_as_tubes=True)
        src = pv.PolyData(mem_slice.points).clean(point_merging=True, merge_tol=1)
        streams = s.streamlines_from_source(src,
            vectors=f"darcy_{ti}", surface_streamlines=True, interpolator_type="cell",
            max_step_length=0.1, initial_step_length=0.1)
        streams[f"darcy_{ti}"][:,1] = 0
        streams.points[:,1] += 1e-4
        pl.add_mesh(streams, color="black", render_lines_as_tubes=True,
                line_width=3, show_scalar_bar=False)
        arrows = streams.glyph(scale=None, orient=f"darcy_{ti}", tolerance=0.05,
                geom=get_triangle("y"), factor=3)
        pl.add_mesh(arrows, color="black",# cmap=cmapinf, scalars=f"GlyphVector", clim=(0, 1e-7),
                show_scalar_bar=False)#, opacity=[0.4,1])

    pl.add_mesh(vw, color="crimson", opacity=0.6)
    pl.camera_position = 'xy'
    pl.camera.zoom(1.4)
    pl.camera.azimuth += 290
    pl.camera.elevation += 30
    pl.screenshot(folder + f"/stacked_{ti}.png", transparent_background=True)
    pl.export_html(folder + f"/stacked_{ti}.html")
    colfig, colax = create_colorbar(cmap, clim, orientation="horizontal",
            figsize=(3, 1.0), right=0.9, bottom=0.6, label="pressure (Pa)")
    colfig.savefig(f"{folder}/colorbar_{ti}_horizontal_{absmax:.2f}.png", transparent=True, dpi=300)

    colfig, colax = create_colorbar(cmap, clim, orientation="vertical",
            figsize=(1.0,3), right=0.3, bottom=None, label="pressure (Pa)")
    colfig.savefig(f"{folder}/colorbar_{ti}_vertical_{absmax:.2f}.png", transparent=True, dpi=300)
