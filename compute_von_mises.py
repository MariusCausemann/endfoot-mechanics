from fenics import *
import numpy as np
from biotbiot.utils import load_data, load_mesh, meshFunction_to_DG, update_material_params
import sys
from biotbiot.utils import von_mises_projector
import yaml


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    with open(f"config_files/{sim_name}.yml") as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    fluid_comp_ids = sum(mesh_config["fluid_comp_ids"].values(), [])

    T = config["T"]
    num_steps = config["num_steps"]
    dt = T/num_steps
    times = np.linspace(0, T, num_steps)
    mesh, sm, bm = load_mesh(mesh_name)
    var_space_map = {"d": VectorFunctionSpace(mesh, "CG", 2)}
    results = load_data(mesh, sim_name, mesh_name, range(num_steps), var_space_map)
    d_series = results["d"]

    cell_ids = sum(mesh_config["porous_cell_ids"].values(), [])
    pvs_ids = mesh_config["fluid_comp_ids"]["pvs"]
    ecs_ids = mesh_config["fluid_comp_ids"]["ecs"]
    mat_params = config["material_parameters"]
    update_material_params(mat_params)

    param_functions = {}
    for param in ["lmbda","mu_s"]:
        param_map = dict(
                zip(
                    cell_ids + pvs_ids + ecs_ids,
                    [mat_params["intracellular"][param]] * len(cell_ids)
                    + [mat_params["pvs"][param]] * len(pvs_ids)
                    + [mat_params["extracellular"][param]] * len(ecs_ids),
                )
            )
        dg_param_func = meshFunction_to_DG(sm, param_map)

        param_functions[param] = dg_param_func

    proj = von_mises_projector(d_series[0], param_functions["lmbda"],
                               param_functions["mu_s"])
    filename = f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_vonmises"
    with HDF5File(mesh.mpi_comm(), filename + ".hdf" , "w") as hdf_file:
        with XDMFFile(filename + ".xdmf") as xdmf_file:
            for t, d in zip(times, d_series):
                stress = proj.project(d)
                stress.rename("vm", "vm")
                hdf_file.write(stress, "vm", t)
                xdmf_file.write(stress, t)



