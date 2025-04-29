import sys
sys.path.append('.')

from plotting.utils import load_results, set_plotting_defaults, efcolor, pvscolor, ecscolor,dpi
 
import matplotlib.pyplot as plt
import numpy as np

def get_diameter(results):
    lumen_diam = results["vessel_diam"] 
    length = max(results["y_coordinates"]) - min(results["y_coordinates"])
    vessel_vol = results["vessel_volume"]
    pvs_vol = results["pvs_volume"]
    ef_vol = results["ef_volume"]
    ecs_vol = results["ecs_volume"]
    pvs_diam = np.sqrt( (vessel_vol + pvs_vol) / (length *np.pi)) * 2
    ef_diam = np.sqrt((vessel_vol + pvs_vol + ef_vol + ecs_vol) / (length *np.pi)) * 2

    return lumen_diam, pvs_diam, ef_diam


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    set_plotting_defaults()

    results = load_results(mesh_name, sim_name)
    times = results["times"]

    lumen_diam, pvs_diam, ef_diam = get_diameter(results)
    pvs_width = (pvs_diam - lumen_diam) / 2

    fig,ax = plt.subplots(dpi=dpi)
    plt.plot(10*times, ef_diam - ef_diam[0], color=efcolor, label="\u0394EF \u2300")
    plt.plot(10*times, lumen_diam - lumen_diam[0], color="crimson", label="\u0394lumen \u2300")
    plt.plot(10*times, pvs_width - pvs_width[0], color=ecscolor, label="\u0394PVS width ")
    plt.legend(loc='upper center', bbox_to_anchor=(0.4, 1.25), ncol=3, frameon=False, columnspacing=1, handlelength=1.0)
    plt.xlabel("cardiac cycles")
    plt.locator_params(axis='both', nbins=5)
    plt.ylabel("size (µm)")
    ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_diam_change.png")


