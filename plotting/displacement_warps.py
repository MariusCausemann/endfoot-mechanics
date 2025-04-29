import pyvista as pv
import os
import numpy as np
from plotting.utils import pvscolor, ecscolor, occolor, read_fenics_results
from cmap import Colormap

sim_name = "cardiac-closed"
mesh_name = "artnew"
scale_fac = 20
time_indices =  [60, 68, 75, 82]

folder = f"results/{mesh_name}_{sim_name}/warp"
os.makedirs(folder, exist_ok=True)
grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
variables=["d"], time_indices=time_indices)
num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]
ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]

domain_length = grid.bounds[3] - grid.bounds[2]
tol = 0.05*domain_length
slices = grid.slice_along_axis(n=3, axis="y", generate_triangles=False, tolerance=tol)

facets = pv.read(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf").scale(1e6)
vw = facets.extract_cells(facets["boundaries"]==mesh_config["facet_ids"]["arterial_wall_id"])
outerwall = facets.extract_cells( np.isin(facets["boundaries"],
                                [mesh_config["facet_ids"]["ecs_outer_id"],
                                mesh_config["facet_ids"]["endfeet_neck_outer_id"],
                                mesh_config["facet_ids"]["cells_outer_id"]
                                ]))

for i,s in enumerate(slices):
    camera = None
    vws = vw.slice(normal="y", origin=s.center - np.array([0,1e-1, 0]))
    outer = outerwall.slice(normal="y", origin=s.center + np.array([0,1e-1, 0]))
    for ti in time_indices:
        s.points += s[f"d_{ti}"] * 1e6 *scale_fac
        pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))
        pl.add_mesh(s.extract_cells(np.isin(s["subdomains"],pvs_ids)), color=pvscolor)
        pl.add_mesh(s.extract_cells(np.isin(s["subdomains"],ecs_ids)), color=ecscolor)
        pl.add_mesh(s.extract_cells(np.isin(s["subdomains"],oc_ids)), color=occolor)
        for cid, col in zip(ef_ids, Colormap("colorbrewer:brbg_6").iter_colors(6)):
            if cid in s["subdomains"]:
                pl.add_mesh(s.extract_cells(np.isin(s["subdomains"],cid)), color=col.rgba)
        pl.add_mesh(vws, color="crimson", line_width=4)
        pl.add_mesh(outer, color="black", line_width=4)
        if camera: pl.camera = camera
        else:
            pl.camera_position = 'xz'
            pl.camera.zoom(1.3)
            camera = pl.camera.copy()
        pl.screenshot(folder + f"/s_{i}_{ti}.png")
        s.points -= s[f"d_{ti}"] * 1e6 *scale_fac


