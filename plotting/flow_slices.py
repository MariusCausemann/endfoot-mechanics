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

ax2num = {"x":0, "y":1, "z":2}
scale_fac = 20
with open(f"config_files/{sim_name}.yml") as conf_file:
        config = yaml.load(conf_file, Loader=yaml.FullLoader)

time_indices =  [config["num_steps"] -i for i in [0, 8, 15, 22, 30]]
time_indices.sort()
print(time_indices)
cmap = Colormap("bids:inferno").to_matplotlib()

grid, mesh_config, config = read_fenics_results(sim_name, mesh_name, 
    variables=["d", "darcy"], time_indices=time_indices)
num_steps = config["num_steps"]
pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
oc_ids = mesh_config["porous_cell_ids"]["outer_cell"]
ef_ids = mesh_config["porous_cell_ids"]["astrocyte"]
cell_ids = oc_ids + ef_ids
domain_length = grid.bounds[3] - grid.bounds[2]
tol = 0.05*domain_length
times = np.linspace(0, config["T"], config["num_steps"])

ecs = grid.extract_cells(np.isin(grid["subdomains"], ecs_ids + pvs_ids))
ics = grid.extract_cells(np.isin(grid["subdomains"], cell_ids))

facets = pv.read_meshio(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf").scale(1e6)
vw = facets.extract_cells(facets["boundaries"]==mesh_config["facet_ids"]["arterial_wall_id"])

if cmax is None:
    absmax = max(minmax([(np.linalg.norm(grid[f"darcy_{ti}"], axis=1)) for ti in time_indices], percentile=95))
    clim = (0, + absmax)
    cmax = "perc99"
else: 
    clim = (0, cmax*1e-6)
    cmax = f"{cmax:.1f}"
    
folder = f"results/{mesh_name}_{sim_name}/flow_slices_{cmax}"
os.makedirs(folder, exist_ok=True)
def get_cut(ecs, ics, vw, cuttype, axis):
    ecs_slices = ecs.slice_along_axis(n=3, axis=axis, generate_triangles=True)
    if axis!="y":ecs_slices=[ecs_slices[1]]
    if cuttype=="slice":
        ecs_cut = ecs_slices
        ics_cut = [ics.slice(normal=axis, origin=s.center, generate_triangles=True) for s in ecs_slices]
        vws = [vw.slice(normal=axis, origin=s.center) for s in ecs_slices]
    elif cuttype=="clip":
        ecs_cut = [ecs.clip(normal=axis, origin=s.center) for s in ecs_slices]
        ics_cut = [ics.clip(normal=axis, origin=s.center) for s in ecs_slices]
        vws = [vw.clip(normal=axis, origin=s.center) for s in ecs_slices]
    return ecs_cut, ics_cut, vws

for cuttype in [ "slice", "clip"]:
    for axis in ["x", "y", "z"]:
        ecs_cut, ics_cut, vws = get_cut(ecs, ics,vw, cuttype, axis)
        ecs_slices,_,_ = get_cut(ecs, ics, vw, "slice", axis)
        for i, (s1, s2, es, v) in enumerate(zip(ecs_cut, ics_cut, ecs_slices,vws)):
            v = v.interpolate(s1)
            camera = None
            for ti in time_indices:
                for g in [s1, s2, es, v]: g.points += g[f"d_{ti}"] * 1e6 *scale_fac
                pl = pv.Plotter(off_screen=True, window_size=(1600, 1600))
                for cid, col in zip(ef_ids, Colormap("colorbrewer:brbg_6").iter_colors(6)):
                    if cid in s2["subdomains"]:
                        pl.add_mesh(s2.extract_cells(np.isin(s2["subdomains"],cid)),
                            color=col.rgba)
                pl.add_mesh(s2.extract_cells(np.isin(s2["subdomains"],oc_ids)), 
                    color=occolor)
                stream = es.streamlines_from_source(pv.PointSet(es.points[::2,:]),
                    vectors=f"darcy_{ti}", surface_streamlines=True, interpolator_type="cell",
                    max_step_length=0.1)
                stream[f"darcy_{ti}"][:,ax2num[axis]] = 0
                pl.add_mesh(stream, scalars=f"darcy_{ti}", clim=clim,
                    cmap=cmap, line_width=2, show_scalar_bar=False, opacity=[0.3,0.8,1])
                if cuttype=="clip":
                    stream = s1.streamlines_from_source(pv.PointSet(s1.points[::10,:]),
                        vectors=f"darcy_{ti}", interpolator_type="cell",max_step_length=0.1)
                    pl.add_mesh(stream, scalars=f"darcy_{ti}", clim=clim,
                        cmap=cmap, line_width=2, show_scalar_bar=False, opacity=[0.3,0.8,1])
                pl.add_mesh(v, color="crimson", opacity=0.5 if cuttype=="clip" else 1, line_width=3)
                if camera: 
                    pl.camera = camera
                else:
                    pl.camera_position = {"x":"yz", "y":"xz", "z":"xy"}[axis]
                    pl.camera.zoom(1.3)
                    camera = pl.camera.copy()
                pl.screenshot(folder + f"/{axis}_{cuttype}_{i}_{ti}_{cmax}.png", transparent_background=True)
                for g in [s1, s2, es, v]: g.points -= g[f"d_{ti}"] * 1e6 *scale_fac

colfig, colax = create_colorbar(cmap, np.array(clim)*1e6, orientation="vertical", 
                                extend="max", figsize=(1.0, 3), label="flow velocity (μm/s)")
colfig.savefig(f"{folder}/colorbar_vertical_{cmax}.png", transparent=True, dpi=300)

colfig, colax = create_colorbar(cmap, np.array(clim)*1e6, orientation="horizontal", 
                                extend="max", figsize=(3, 1.0), right=0.9, bottom=0.6, label="flow velocity (μm/s)")
colfig.savefig(f"{folder}/colorbar_horizontal_{cmax}.png", transparent=True, dpi=300)
