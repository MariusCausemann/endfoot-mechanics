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

time_indices =  [60, 68, 75, 82]
cmap = Colormap("bids:inferno").to_matplotlib()
folder = f"results/{mesh_name}_{sim_name}/flow_stack"
os.makedirs(folder, exist_ok=True)
grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["darcy"], time_indices=time_indices)
num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]
ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]
cell_ids = oc_ids + ef_ids
domain_length = grid.bounds[3] - grid.bounds[2]
tol = 0.04*domain_length
times = np.linspace(0, config["T"], config["num_steps"])

ecs = grid.extract_cells(np.isin(grid["subdomains"], ecs_ids + pvs_ids))
ics = grid.extract_cells(np.isin(grid["subdomains"], cell_ids))

ecs_slices = ecs.slice_along_axis(n=3, axis="y", generate_triangles=True, tolerance=tol)
ics_slices = [ics.slice(normal="y", origin=s.center, generate_triangles=True) for s in ecs_slices]
complete_slice = [grid.slice(normal="y", origin=s.center, generate_triangles=True) for s in ecs_slices]
facets = pv.read(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf").scale(1e6)
vw = facets.extract_cells(facets["boundaries"]==mesh_config["facet_ids"]["arterial_wall_id"])

em = pv.read(f"meshes/{mesh_name}/img/raw.vtk").scale(1e-3)
emslice = em.slice(normal="x",origin=[15, 10, 14.0])

def grow(mesh, s):
    m = mesh.copy()
    m.points[:] = (m.points - m.center)*s + m.center
    return m

def filter_streamlines(sl, min_length = 0):
    lines = sl.compute_cell_sizes().split_bodies()
    return pv.merge([l for l in lines if abs(l["Length"]).sum() > min_length], merge_points=False)

for ti in time_indices:
    absmax = max(minmax([(np.linalg.norm(s[f"darcy_{ti}"], axis=1)) for s in ecs_slices], percentile=92))
    clim = (0, absmax)
    colfig, colax = create_colorbar(cmap, np.array(clim)*1e6, orientation="horizontal", extend="max",
                    figsize=(2.2,1), right=None, bottom=0.7, label="flow velocity (μm/s)")
    colfig.savefig(f"{folder}/colorbar_{ti}.png", transparent=True, dpi=300)
    pl = pv.Plotter(off_screen=True, window_size=(2400, 2000))
    pl.add_mesh(vw, color="crimson", opacity=0.6)

    for i,(s1,s2) in enumerate(zip(ecs_slices, ics_slices)):
        for cid, col in zip(ef_ids, Colormap("colorbrewer:brbg_6").iter_colors(6)):
            if cid in s2["subdomains"]:
                pl.add_mesh(s2.extract_cells(np.isin(s2["subdomains"],cid)), color=col.rgba)
        pl.add_mesh(s2.extract_cells(np.isin(s2["subdomains"],oc_ids)), color=occolor)
        stream = s1.streamlines_from_source(pv.PointSet(s1.points[::2,:]),
        vectors=f"darcy_{ti}", surface_streamlines=True, interpolator_type="cell")
        pl.add_mesh(stream, scalars=f"darcy_{ti}", clim=clim,
            cmap=cmap, line_width=2, show_scalar_bar=False, opacity=[0.2,0.8,1])
    source = grow(vw, 1.05)# pv.PointSet(pv.merge(ecs_slices).points[::5,:])
    streams = grid.streamlines_from_source(source ,vectors=f"darcy_{ti}",
                 interpolator_type="cell")
    streams = filter_streamlines(streams, min_length=8)
    pl.add_mesh(streams, scalars=f"darcy_{ti}", clim=clim,
            cmap=cmap, line_width=3, show_scalar_bar=False, opacity=[0.2,0.8,1])
    pl.add_mesh(emslice, scalars="em", show_scalar_bar=False, cmap="Greys_r", 
        clim=(100, 140))
    pl.camera_position = 'xy'
    pl.camera.zoom(1.42)
    pl.camera.azimuth += 290
    pl.camera.elevation += 30
    pl.screenshot(folder + f"/stacked_{ti}.png", transparent_background=True)

    old_pos = pl.camera.position

    detail_plots = [(1, (4.5, 9.5, 12.4)),
                    (0,(12.3, 1, 24))]
    for si, pos in detail_plots:
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        s1,s2 = ecs_slices[si], ics_slices[si]
        for cid, col in zip(ef_ids, Colormap("colorbrewer:brbg_6").iter_colors(6)):
            if cid in s2["subdomains"]:
                pl.add_mesh(s2.extract_cells(np.isin(s2["subdomains"],cid)), color=col.rgba)
        pl.add_mesh(s2.extract_cells(np.isin(s2["subdomains"],oc_ids)), color=occolor)

        stream = s1.streamlines_from_source(pv.PointSet(s1.points[::2,:]),
        vectors=f"darcy_{ti}", surface_streamlines=True,
            interpolator_type="cell", max_step_length=0.1, initial_step_length=0.1)
        pl.add_mesh(stream, scalars=f"darcy_{ti}", clim=clim,
            cmap=cmap, line_width=3, show_scalar_bar=False, opacity=[0.3,0.8,1])

        pl.camera.focal_point = pos
        pl.camera.position = np.array(pos) + np.array((0,7,0))
        pl.camera.roll -=90
        pl.screenshot(folder + f"/detail_{si}_{ti}_zoom.png", transparent_background=True)
        pl.camera_position = "xz"
        pl.screenshot(folder + f"/detail_{si}_{ti}.png", transparent_background=True)

colfig, colax = create_colorbar(cmap, np.array(clim)*1e6, orientation="vertical", 
                                extend="max", figsize=(1.0, 2), label="flow velocity (μm/s)")
colfig.savefig(f"{folder}/colorbar_vertical_{ti}.png", transparent=True, dpi=300)
