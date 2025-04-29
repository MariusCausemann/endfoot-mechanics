from fenics import *
import numpy as np
import yaml
from ufl import nabla_div
from scipy.spatial import cKDTree as KDTree

def as_P0_function(mesh_f):
    '''Represent as DG0'''
    mesh = mesh_f.mesh()
    assert mesh_f.dim() == mesh.topology().dim()
    P0 = FunctionSpace(mesh, 'DG', 0)
    f = Function(P0)
    f.vector().set_local(mesh_f.array())
    return f

def epsilon(u):
    return sym(grad(u))

class von_mises_projector(object):
    def __init__(self, d, lmbda, mu):
        W = d.function_space()
        self.mesh = W.mesh()
        self.V = FunctionSpace(self.mesh, "DG", 0)
        self.v = TestFunction(self.V)
        Pv = TrialFunction(self.V)
        a = inner(self.v, Pv) * dx
        A = assemble(a)
        self.projector = PETScLUSolver(as_backend_type(A), "mumps")
        self.dx = dx
        self.lmbda = lmbda
        self.mu = mu

    def sigma(self, u):
        gdim = self.mesh.geometric_dimension()
        return self.lmbda * nabla_div(u) * Identity(gdim) + 2 * self.mu * epsilon(u)

    def project(self, d):
        gdim = self.mesh.geometric_dimension()
        s = self.sigma(d) - (1.0 / 3) * tr(self.sigma(d)) * Identity(
            gdim
        )  # deviatoric stress
        von_mises = sqrt(3.0 / 2 * inner(s, s))
        FF = assemble(von_mises * self.v * self.dx)
        f = Function(self.V)
        self.projector.solve(f.vector(), FF)
        return f


def write_to_file(results, time, names, hdf5_file, xdmf_file):
    for k, r in enumerate(results):
        r.rename(names[k], names[k])
        xdmf_file.write(r, time)
        hdf5_file.write(r, r.name(), time)


def meshFunction_to_DG(mf, mf_to_dg_value_map=dict()):
    V = FunctionSpace(mf.mesh(), "DG", 0)
    v = Function(V)
    x = V.tabulate_dof_coordinates()
    for i in range(x.shape[0]):
        mf_val = mf.array()[i]
        if mf_val in mf_to_dg_value_map.keys():
            v.vector().vec().setValueLocal(i, mf_to_dg_value_map[mf_val])
    return v


def create_dirichlet_bcs(boundary_conditions, H):
    bcs = []
    for bc in boundary_conditions:
        marker_id, subspace_id, bc_val, boundary_marker = bc
        if isinstance(subspace_id, int):
            bc_d = DirichletBC(H.sub(subspace_id), bc_val, boundary_marker, marker_id)
        elif len(subspace_id) == 2:
            bc_d = DirichletBC(
                H.sub(subspace_id[0]).sub(subspace_id[1]),
                bc_val,
                boundary_marker,
                marker_id,
            )
        bcs.append(bc_d)
    return bcs

class update_fem_func(object):
    def __init__(self, fem_func, args, func):
        self.fem_func = fem_func
        self.args = args
        self.func = func

    def update(self, t, results):
        self.fem_func.vector()[:] = self.func(t, results, self.args)

def compute_midline_normal(mesh_points, midline_points):
    tree = KDTree(midline_points)
    d, idx = tree.query(mesh_points)
    closest_points = midline_points[idx]
    vecs = closest_points - mesh_points
    vecs[:,1] = 0.0
    normed_vec = vecs / np.linalg.norm(vecs, axis=1)[:, np.newaxis]
    return normed_vec

def get_lumen_centerline_direction(fenics_mesh, mesh_name):
    with open(f"meshes/{mesh_name}/centerline.yml", 'r') as file:
        centerline = np.array(yaml.load(file, Loader=yaml.FullLoader))

    V = VectorFunctionSpace(fenics_mesh, "CG", 1)
    n = Function(V)
    points = fenics_mesh.coordinates()
    normal = compute_midline_normal(points, centerline)
    d_to_vert = dof_to_vertex_map(V)
    n.vector().set_local(normal.flatten()[d_to_vert])
    as_backend_type(n.vector()).vec().ghostUpdate()
    return n


class InterfaceExpression(UserExpression):
    def __init__(self, mesh, bm, marker_value_dict, **kwargs):
        super().__init__(**kwargs)
        self.mesh = mesh
        self.bm = bm
        self.marker_value_dict = marker_value_dict

    def eval_cell(self, values, x, ufc_cell):
        cell = Cell(self.mesh, ufc_cell.index)
        f = list(facets(cell))[ufc_cell.local_facet]
        values[0] = self.marker_value_dict[self.bm[f]]
    
    def value_shape(self):
        return ()
    

def update_material_params(material_parameter):
    mat_dicts = [material_parameter["intracellular"]]
    mat_dicts += [material_parameter["extracellular"]]
    mat_dicts += [material_parameter["pvs"]]

    for mat_params in mat_dicts:
        E = mat_params["E"]
        nu = mat_params["nu"]
        mat_params["mu_s"] = E / (2.0 * (1.0 + nu))
        mat_params["lmbda"] = nu * E / ((1.0 - 2.0 * nu) * (1.0 + nu))
        if not "kappa" in mat_params.keys():
            D_p = mat_params["D_p"]
            mat_params["kappa"] = (
                D_p * (1 - 2 * nu) * (1 + nu) / (E * (1 - nu)) * mat_params["mu_f"]
            )
        mat_params["K"] = mat_params["kappa"] / mat_params["mu_f"]
