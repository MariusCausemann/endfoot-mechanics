import pyvista as pv 
from fenics import *
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.append("meshing")
sys.path.append(".")
from mark_arteriole_boundaries import mark_boundaries
from evaluate_mesh_deformation import evaluate_gap_sizes

ecs_id = 1
pvs_id = 0
pvs_width = 300*1e-9
lumen_id = 473
z_rotation = 0
x_rotation = 0

def smooth(grid, iter=1):
    for i in range(iter):
        grid = grid.ptc().ctp()
    return grid



def prepare(grid, lumen_id):
    lumen = grid.extract_cells(grid["label"]==lumen_id)
    grid = grid.extract_cells(grid["label"]!=lumen_id).extract_largest()
    grid["subdomains"] = grid["label"]
    gs, vol = evaluate_gap_sizes(grid, [ecs_id], set(grid["label"]) - set([ecs_id]))
    lumsurf = lumen.extract_surface()
    grid.compute_implicit_distance(lumsurf, inplace=True)
    grid["implicit_distance"] = abs(grid["implicit_distance"])
    ecs = grid.extract_cells(grid["label"]==ecs_id)
    ecs["gap_size"] = gs
    ecs = ecs.point_data_to_cell_data()
    ecs = smooth(ecs, iter=12).ptc()
    pvs_marker = np.logical_or(ecs["gap_size"] > 400.0e-9, ecs["implicit_distance"] < pvs_width)
    ind = np.where(grid["label"]==ecs_id)[0][np.where(pvs_marker)[0]]
    grid["label"][ind] = pvs_id
    return grid, lumen

grid = pv.read(f"meshes/artnewref/mesh.xdmf")
grid.rotate_z(z_rotation, inplace=True)
grid.rotate_x(x_rotation, inplace=True)

grid.points *= 1e-9 #nm to m
grid, lumen = prepare(grid, lumen_id)


pv.save_meshio(f"meshes/artnewref/artnewref.xdmf", grid)
slices = lumen.slice_along_axis(n=100, axis="y")
midpoints = np.array([sl.center_of_mass() for sl in slices])
centerline = pv.PolyData(midpoints)
with open(f"meshes/artnew/centerline.yml", 'w') as file:
    yaml.dump(centerline.points.tolist(), file)

mesh = Mesh()
with XDMFFile(f"meshes/artnewref/artnewref.xdmf") as mf:
    mf.read(mesh)
    sm = MeshFunction("size_t", mesh, 3, 0)
    mf.read(sm, "label")
    sm.rename("subdomains","")

cell_ids = set(np.unique(grid["label"]))
cell_ids.remove(pvs_id)
cell_ids.remove(ecs_id)

bm = mark_boundaries(mesh, sm, cell_ids, eps=0.01)

bm.rename("boundaries","")

with XDMFFile(f"meshes/artnewref/artnewref_facets.xdmf") as outfile:
    outfile.write(bm)

with XDMFFile(f"meshes/artnewref/artnewref.xdmf") as outfile:
    outfile.write(sm)

mesh_dict = {"facet_ids":
                {"pvs_inlet_id" : 10,
                "pvs_outlet_id" : 11,
                "endfeet_neck_outer_id" : 9,
                "endfeet_axial_id" : 7,
                "arterial_wall_id" : 4,
                "interf_id" : 12,
                "aqp_membrane_id" : 13,
                "cells_outer_id" :  9,
                "cells_axial_id" : 7,
                "ecs_axial_id" : 6,
                "ecs_outer_id" : 8,
                "string_id" : 5},
            "porous_cell_ids":{"astrocyte":[int(c) for c in cell_ids]},
            "fluid_comp_ids" :{"pvs":[pvs_id], "ecs":[ecs_id]}
            }

with open(f"config_files/artnewref.yml", "w") as mesh_config:
    yaml.dump(mesh_dict, mesh_config)