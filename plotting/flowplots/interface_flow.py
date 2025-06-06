
import sys 
sys.path.append('.')
sys.path.append('..')

from plotting.utils import load_results, m3tofl, dpi, set_plotting_defaults, efcolor, pvscolor, ecscolor
 
import matplotlib.pyplot as plt
set_plotting_defaults()

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
times = results["times"]
aqp_flow = results["aqp_membrane_flow"]
non_aqp_flow = results["non_aqp_membrane_flow"]

fig,ax = plt.subplots(dpi=dpi)
plt.plot(10*times, aqp_flow*m3tofl, color="lightgreen", label="high AQP4")
plt.plot(10*times, non_aqp_flow*m3tofl, color=efcolor, label="low AQP4")
plt.xlabel("cardiac cycles")
plt.ylabel("flow ($\mu m^3$/s)")
plt.locator_params(axis='both', nbins=5)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_interface_flow.png")