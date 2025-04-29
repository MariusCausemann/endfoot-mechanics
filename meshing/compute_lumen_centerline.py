import numpy as np
import pyvista as pv
import sys
import yaml

def compute_lumen_centerline(mesh, lumen):
    surf = mesh.extract_surface()
    surf.compute_normals(non_manifold_traversal=False, inplace=True)
    b = np.array(lumen.bounds).reshape(3,2)
    xrng = np.linspace(*b[0], 100)
    yrng = np.linspace(*b[1], 100)
    zrng = np.linspace(*b[2], 100)
    grid = pv.RectilinearGrid(xrng, yrng, zrng)
    grid.compute_implicit_distance(surf, inplace=True)
    vessel = grid.threshold(0, invert=True).extract_largest()
    slices = vessel.slice_along_axis(n=100, axis="y")
    midpoints = np.array([sl.center_of_mass() for sl in slices])
    midline = pv.PolyData(midpoints)
    return midline

if __name__ == '__main__':
    mesh_name = sys.argv[1]
    mesh = pv.read(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    lumen = pv.read(f"meshes/{mesh_name}/surfaces/lumen.stl").extract_geometry()
    lumen.points *= 1e-9
    centerline = compute_lumen_centerline(mesh, lumen)
    centerline.save(f"meshes/{mesh_name}/centerline.vtk")

    with open(f"meshes/{mesh_name}/centerline.yml", 'w') as file:
        yaml.dump(centerline.points.tolist(), file)