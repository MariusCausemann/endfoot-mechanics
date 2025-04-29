from dolfin import *
from petsc4py import PETSc
from pathlib import Path
import numpy as np
from biotbiot.biotBiotFluxWeakForm import (
    create_function_spaces,
    create_measures,
    biot_biot_system,
)
from biotbiot.utils import write_to_file, create_dirichlet_bcs, von_mises_projector, as_P0_function

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["cpp_optimize_flags"] = "-O2 -ftree-vectorize"


def solve(
    mesh,
    T,
    num_steps,
    material_parameter,
    boundary_marker,
    subdomain_marker,
    boundary_conditions,
    filename,
    time_dep_expr=(),
    g=Constant(0),
    f=None,
    robin_boundaries=(),
    interface_id=1,
    degree=2,
):
    porous_id = 1
    fluid_id = 2

    names = ["d", "darcy", "p", "vm", "divd"]

    time = 0.0
    dt = Constant(T / num_steps)

    with HDF5File(
        mesh.mpi_comm(), f"{Path(filename).parent}/{Path(filename).stem}.hdf", "w"
    ) as hdf5_file:
        xdmf_file = XDMFFile(f"{Path(filename).parent}/{Path(filename).stem}.xdmf")
        xdmf_file.parameters["functions_share_mesh"] = True
        xdmf_file.parameters["rewrite_function_mesh"] = False

        measures = create_measures(
            mesh, subdomain_marker, boundary_marker, porous_id, fluid_id, interface_id
        )
        H = create_function_spaces(mesh, degree)
        lhs, rhs, sol = biot_biot_system(
            mesh,
            material_parameter,
            H,
            measures,
            dt,
            robin_boundaries=robin_boundaries,
            f=f,
            g=g,
        )

        bcs = create_dirichlet_bcs(boundary_conditions, H)
        A = assemble(lhs)
        for bc in bcs:
            bc.apply(A)
        solver = PETScLUSolver(as_backend_type(A), "mumps")
        d,q,p = list(sol.split(deepcopy=True))
        vm_proj = von_mises_projector(
            d, material_parameter["lmbda"], material_parameter["mu_s"]
        )
        PETSc.Sys.Print("VM projection...")
        vm = vm_proj.project(d)
        p0_sm = as_P0_function(subdomain_marker)
        PETSc.Sys.Print("div d projection...")
        divd = project(div(d), FunctionSpace(mesh, "DG", 0), 
                    solver_type="cg", preconditioner_type="hypre_amg")
        write_to_file([d, q, p, vm, divd, p0_sm], time, names + ["subdomains"], hdf5_file, xdmf_file)
        for i in range(num_steps):
            time = (i + 1) * dt.values()[0]
            for expr in time_dep_expr:
                expr(time, sol)
            FF = assemble(rhs)
            for bc in bcs:
                bc.apply(FF)
            PETSc.Sys.Print("solving...")
            solver.solve(sol.vector(), FF)
            d,q,p = list(sol.split(deepcopy=True))
            vm = vm_proj.project(d)
            divd = project(div(d), FunctionSpace(mesh, "DG", 0), 
                    solver_type="cg", preconditioner_type="hypre_amg")
            write_to_file([d, q, p, vm, divd], time, names, hdf5_file, xdmf_file)
        xdmf_file.close()
    return sol
