import sys
sys.path.append('.')
from plotting.utils import load_results, set_plotting_defaults, efcolor, pvscolor, ecscolor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
set_plotting_defaults()

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)

y_coords = results["y_coordinates"]
pvs_area = results["pvs_area"] 
vessel_area = results["vessel_area"] 
ef_area = results["ef_area"]

A2D = lambda a: 2*np.sqrt(a/np.pi)

vessel_diam = A2D(vessel_area)
ef_diam_inner = A2D(vessel_area + pvs_area)
pvs_width = ef_diam_inner - vessel_diam
ef_diam_outer = A2D(vessel_area + pvs_area + ef_area)

sns.set_context("talk")
plt.figure( dpi=500)
plt.plot(y_coords,  pvs_width, ".-", color=pvscolor, label="PVS width")
plt.plot(y_coords, ef_diam_inner, ".-", color=efcolor, label="EF \u2300")
plt.plot(y_coords, vessel_diam , ".-", color="firebrick", label="lumen \u2300")

plt.xlabel("vessel length ($\mu$m)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, columnspacing=0.5,
            handlelength=.5, frameon=False)
plt.ylabel(r"size ($\mu m$)")
#plt.ylabel(r"cross section area [$\mu m^2$]")
plt.ylim(0,18)
if __name__ == '__main__':
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_cross_section.png")
