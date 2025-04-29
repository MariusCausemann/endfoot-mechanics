import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml
import os
from IPython import embed
from fenics import *
from tqdm import tqdm
#from cache_to_disk import cache_to_disk

m3topl = 1e15
m3tofl = 1e18
sim_name = None
mesh_name = None
abs_path = os.path.dirname(os.path.abspath(__file__))

dpi = 500
efcolor="darkgreen"
pvscolor="steelblue"
ecscolor="black"
occolor="gray"

def set_plotting_defaults():
    import dufte
    import seaborn as sns
    for k,v in dufte.style.items():
        if "color" in k:
            dufte.style[k] = "black"
    plt.style.use(dufte.style)
    sns.set_context("poster")
    mpl.rcParams["figure.subplot.left"] = 0.25
    mpl.rcParams["figure.subplot.bottom"] = 0.25
    mpl.rcParams["figure.subplot.top"] = 0.85
    mpl.rcParams["legend.columnspacing"] = 1.0
    mpl.rcParams["legend.handlelength"] = 1.2
    mpl.rcParams["savefig.transparent"] = True

def create_colorbar(cmap, clim, figsize=(0.2, 3), right=0.3, bottom=None, **kwargs):
    fig, ax = plt.subplots(figsize=figsize)
    norm = mpl.colors.Normalize(vmin=clim[0], vmax=clim[1])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax, **kwargs)
    fig.subplots_adjust(right=right, bottom=bottom)
    return fig, ax

def draw_brace(ax, xspan, yy, text, color="black"):
    """Draws an annotated brace on the axes."""
    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color=color, lw=2)

    ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom')

def load_results(mesh_name, sim_name):
    results = {}
    with open(f"{abs_path}/../results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_ts.yml", 'r') as file:
        results = yaml.load(file, Loader=yaml.FullLoader)

    with open(f"{abs_path}/../results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_pyvista.yml", 'r') as file:
        results.update(yaml.load(file, Loader=yaml.FullLoader))

    #with open(f"{abs_path}/../results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_peak_distr.yml", 'r') as file:
    #    results.update(yaml.load(file, Loader=yaml.FullLoader))

    with open(f"{abs_path}/../results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_flow_metrics.yml", 'r') as file:
        results.update(yaml.load(file, Loader=yaml.FullLoader))


    with open(f"{abs_path}/../results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_disp_stress_metrics.yml", 'r') as file:
        results.update(yaml.load(file, Loader=yaml.FullLoader))

    for k,v in results.items():
        if isinstance(v, list):
            results[k] = np.array(v)

    return results


def gap_size_histo(counts, binpoints, times):  
    bp = binpoints #[30:160]
    cts = counts #[:,30:160]
    y, x = np.meshgrid(bp, times)
    cts /= cts.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots()#subplot_kw={"projection": "3d"})
    c = ax.pcolormesh(x, y, cts, cmap=cmocean.cm.algae, rasterized=True)
    #c = ax.plot_surface(x, y, counts, cmap=cmocean.cm.algae)
    f = 3 / times[-1]
    m = np.argmax(cts, axis=1)
    peaks = bp[m]
    mean = (cts * bp * len(bp)).mean(axis=1)
    #plt.plot(times, peaks, c="red", marker="*")
    plt.plot(times, mean, c="yellow", marker="*", label="mean")
    plt.plot(times, bp[0] - 3 + np.sin(2*np.pi*f*times), label="vessel pulsation")
    cbar = fig.colorbar(c)
    cbar.set_label('relative frequency')
    plt.legend(loc="upper right")
    plt.xlabel("time (s)")
    plt.ylabel("local gap size [nm]")

def line_data_over_time(line_data, times, coords, cmap):
    y, x = np.meshgrid(coords, times)
    fig, ax = plt.subplots()
    #cmap=cmocean.cm.thermal
    c = ax.pcolormesh(x, y, line_data, cmap=cmap, rasterized=True)
    cbar = fig.colorbar(c)
    plt.xlabel("time (s)")
    plt.ylabel(r"x [$\mu$m]")
   
#@cache_to_disk(1)
def read_fenics_results(sim_name, mesh_name, variables, time_indices):
    from vtk_adapter import create_vtk_structures
    import pyvista as pv

    mesh = Mesh()
    infile =  XDMFFile(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    infile.read(mesh)
    gdim = mesh.geometric_dimension()
    sm = MeshFunction("size_t", mesh, gdim)
    infile.read(sm, "subdomains")
    infile.close()

    Uh = VectorElement("CR", mesh.ufl_cell(), 1)
    Vh = FiniteElement("RT", mesh.ufl_cell(), 1)
    Ph = FiniteElement("DG", mesh.ufl_cell(), 0)
    H = FunctionSpace(mesh, MixedElement([Uh, Vh, Ph]))
    DG0 = FunctionSpace(mesh, "DG", 0)
    DG1 = FunctionSpace(mesh, "DG", 1)
    VDG1 = VectorFunctionSpace(mesh, "DG", 1)
    spaces = {"d":H.sub(0).collapse(), "darcy":H.sub(1).collapse(),
              "p":H.sub(2).collapse(), "vm":DG0, "divd":DG0}
    if "darcy" in variables:
        target_spaces = {"d":VDG1, "darcy":VDG1}
        structspace = DG1
    else:
        target_spaces = {"d":VectorFunctionSpace(mesh, "CG", 1)}
        structspace = FunctionSpace(mesh, "CG", 1)

    infile = HDF5File(mesh.mpi_comm(), f"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.hdf", "r")
    results = {v:[] for v in variables}
    for v in variables:
        for ti in tqdm(time_indices):
            f = Function(spaces[v])
            infile.read(f,f"{v}/vector_{ti}")
            if v in target_spaces.keys():
                results[v].append(interpolate(f, target_spaces[v]))
            else:
                results[v].append(f)
    infile.close()

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    with open(f"config_files/{sim_name}.yml") as conf_file:
            config = yaml.load(conf_file, Loader=yaml.FullLoader)

    topology, cell_types, x = create_vtk_structures(structspace)
    grid = pv.UnstructuredGrid(topology, cell_types, x)
    for v, res in results.items():
        print(v, spaces[v], )
        for ti,r in zip(time_indices, res):
            if len(r.ufl_shape) > 0:
                grid[f"{v}_{ti}"] = r.vector()[:].reshape(-1, 3)
            else:
                grid[f"{v}_{ti}"] = r.vector()[:]
    
    grid["subdomains"] = sm.array()[:]
    grid.scale(1e6, inplace=True)

    return grid, mesh_config, config