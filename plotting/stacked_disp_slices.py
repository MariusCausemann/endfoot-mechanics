from mpi4py import MPI
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

time_indices =  [68]
folder = f"results/{mesh_name}_{sim_name}/disp_stack"
os.makedirs(folder, exist_ok=True)
grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["d"], time_indices=time_indices)
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

ecs_slices = ecs.slice_along_axis(n=3, axis="y", tolerance=tol)
ics_slices = [ics.slice(normal="y", origin=s.center) for s in ecs_slices]
ics_x = ics.slice(normal=[0.8,0,-0.2])
facets = pv.read_meshio(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf").scale(1e6)
vw = facets.extract_cells(facets["boundaries"]==mesh_config["facet_ids"]["arterial_wall_id"])

comp_slices = [grid.slice(normal="y", origin=s.center) for s in ecs_slices]
comp_x = grid.slice(normal=[0.8,0,-0.2])

#em = pv.read(f"meshes/{mesh_name}/img/raw.vtk").scale(1e-3)
#emslice = em.slice(normal="x",origin=[15, 10, 14.0])

def make_arrows(sl, vec, factor, grid):
    n = sl.compute_normals()["Normals"].mean(axis=0)
    print(n)
    p = pv.Plane(center=sl.center, direction=n, i_size=20,j_size=20,
        i_resolution=12,j_resolution=12).sample(grid, tolerance=0.3)
    arrows = p.glyph(orient=vec, scale=vec, factor=factor)
    return arrows

cmap=Colormap("cmasher:fall_r").to_matplotlib()
for ti in time_indices:
    pl = pv.Plotter(off_screen=True, window_size=(2000, 2000))
    pl.add_mesh(vw, color="crimson", opacity=0.6)
    clim = minmax((np.linalg.norm(s[f"d_{ti}"], axis=1)))
    for s in comp_slices + [comp_x]:
        arr = make_arrows(s, f"d_{ti}", 4e7, grid)
        pl.add_mesh(arr, color="black")
        pl.add_mesh(s, scalars=f"d_{ti}", show_scalar_bar=False,clim=clim,
            cmap=cmap) #solar_r or "turbid" ?
        mem = get_surface_interface_line(s.clean(), pvs_ids + ecs_ids, cell_ids)
        pl.add_mesh(mem, color="black", line_width=3, render_lines_as_tubes=True)
    
    #pl.add_arrows(cent=np.array([14, 25, 14]), direction=np.array([0.25,0,0.75]),
    #              mag=3, show_scalar_bar=False, color="black")
    #pl.add_mesh(emslice, scalars="em", show_scalar_bar=False, cmap="Greys_r", clim=(110, 140))
    pl.camera_position = 'xy'
    pl.camera.focal_point = np.array(grid.center) - np.array([0,1,0])
    pl.camera.zoom(1.42)
    pl.camera.azimuth += 290
    pl.camera.elevation += 30
    pl.screenshot(folder + f"/stacked_{ti}.png", transparent_background=True)
    colfig, colax = create_colorbar(cmap, np.array(clim)*1e9, orientation="vertical", 
                                extend=None, figsize=(1.0, 2), label="displacement (nm)")
    colfig.savefig(f"{folder}/colorbar_{ti}.png", transparent=True, dpi=300)



"""
for s in ics_slices + [ics_x]:
    for cid, col in zip(ef_ids, Colormap("colorbrewer:brbg_6").iter_colors(6)):
        if cid in s["subdomains"]:
            pl.add_mesh(s.extract_cells(np.isin(s["subdomains"],cid)), 
                            color=col.rgba)
    pl.add_mesh(s.extract_cells(np.isin(s["subdomains"],oc_ids)), color=occolor)
"""
from IPython import embed; embed()
