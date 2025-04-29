from dolfin import *

def eps(u):
    return sym(grad(u))

def create_function_spaces(mesh, degree=2):
    Uh = VectorElement("CR", mesh.ufl_cell(), 1)
    Vh = FiniteElement("RT", mesh.ufl_cell(), degree - 1)
    Ph = FiniteElement("DG", mesh.ufl_cell(), degree - 2)
    H = FunctionSpace(mesh, MixedElement([Uh, Vh, Ph]))
    return H


def create_measures(mesh, subdomain_marker, boundary_marker, ics_id, ecs_id, interf_id):
    dxi = Measure(
        "dx", domain=mesh, subdomain_data=subdomain_marker, subdomain_id=ics_id
    )
    dxe = Measure(
        "dx", domain=mesh, subdomain_data=subdomain_marker, subdomain_id=ecs_id
    )
    dxD = Measure("dx", domain=mesh, subdomain_data=subdomain_marker)
    dS = Measure("dS", domain=mesh, subdomain_data=boundary_marker)
    ds = Measure("ds", domain=mesh, subdomain_data=boundary_marker)
    ds_Sig = dS(interf_id)

    return [dxi, dxe, dxD, ds, ds_Sig]


def get_lhs(d, q, p, w, v, z, dt, dx, dS, mesh, material_parameter, eta):
    mu = material_parameter["mu_s"]
    lmbda = material_parameter["lmbda"]
    alpha = material_parameter["alpha"]
    c = material_parameter["c"]
    K = material_parameter["K"]

    n = FacetNormal(mesh)
    h = CellDiameter(mesh)
    h_avg = avg(h)

    normal = lambda u: dot(u, n) * n
    tangential = lambda u: u - normal(u)
    AVG = lambda tau: (tau("+") * n("+") - tau("-") * n("-")) / 2
    H = d.ufl_operands[0].ufl_operands[0].ufl_function_space()
    assert assemble(Constant(1)*dS) > 0
    
    if H.sub(0).ufl_element().family() == "Crouzeix-Raviart": # Hansbo, 2003 http://www.numdam.org/article/M2AN_2003__37_1_63_0.pdf
        ah = lambda u, w, mu: (
            mu*inner(eps(u), eps(w)) * dx
            + avg(mu)*eta / h_avg * inner(jump(u), jump(w)) * dS
            #+ mu*eta / h * inner(u, w) * ds
        )

    if H.sub(0).ufl_element().family() == "Raviart-Thomas":
        ah = lambda u, w, mu: (
            mu*inner(eps(u), eps(w)) * dx
            - avg(mu)*inner(AVG(eps(u)), jump(tangential(w))) * dS
            - avg(mu)*inner(AVG(eps(w)), jump(tangential(u))) * dS
            + avg(mu)*eta / h_avg * inner((jump(tangential(u))), jump(tangential(w))) * dS
        )

    if H.sub(0).ufl_element().family() == "Lagrange":
        ah = lambda u, w, mu: mu*inner(eps(u), eps(w)) * dx


    a = 2*ah(d, w, mu) + lmbda * div(d) * div(w) * dx - alpha* p * div(w) * dx
    a += 1.0 / K * inner(q, v) * dx - p * div(v) * dx
    a += alpha * div(d) * z / dt * dx + div(q) * z * dx + c * p * z / dt * dx
    return a

def get_rhs(w, q, f, g, u_n, p_n, dx, dt, material_parameter):
    alpha = material_parameter["alpha"]
    c = material_parameter["c"]
    rhs = inner(f, w) * dx + q * g * dx
    return  rhs + alpha*div(u_n) * q /dt * dx + c*p_n * q /dt * dx


def generate_robin_terms(trial_functions, test_functions, robin_bcs):
    # generate forms for BCs of the type 
    lhs, rhs = 0, 0
    for bc in robin_bcs:
        ds, subspace_id, func_l, func_r = bc
        u, v = trial_functions[subspace_id], test_functions[subspace_id]
        lhs += func_l(u, v) * ds
        rhs -= func_r(v)*ds
    return lhs, rhs


def biot_biot_system(
    mesh,
    material_parameter,
    H,
    measures,
    dt,
    robin_boundaries=(),
    g=Constant(0),
    f=None,
):
    dxi, dxe, dxD, ds, ds_Sig = measures
    dS = Measure("dS", mesh)

    (d, q, p) = TrialFunctions(H)
    (w, v, z) = TestFunctions(H)
    sol = Function(H)
    (d_n, q_n, p_n) = split(sol)

    if f is None:
        f = Constant([0.0] * mesh.geometric_dimension())

    lhs = get_lhs(d, q, p, w, v, z, dt, dxD, dS, mesh, material_parameter, eta=1e0)
    rhs = get_rhs(w, z, f, g, d_n, p_n, dxD, dt, material_parameter)

    L_p = material_parameter["L_p"]

    n = FacetNormal(mesh)("+")
    interface_term = 1/L_p * inner(avg(q), n) * (inner(avg(v), n)) * ds_Sig
    robin_lhs, robin_rhs = generate_robin_terms((d, q, p), (w, v, z), robin_boundaries)
    
    lhs += robin_lhs  + interface_term
    rhs += robin_rhs
    return lhs, rhs, sol
