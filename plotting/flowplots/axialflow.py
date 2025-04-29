
import sys 
sys.path.append('.')
sys.path.append('..')

from plotting.utils import load_results, m3topl, set_plotting_defaults, efcolor, pvscolor, ecscolor, dpi
import matplotlib.pyplot as plt
set_plotting_defaults()

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
times = results["times"]
inflow = results["inlet_flow"]
outflow = results["outlet_flow"]

plt.figure(dpi=dpi)
plt.plot(times, inflow*m3topl, label="pvs inlet")
plt.plot(times, outflow*m3topl, label="pvs outlet")
plt.xlabel("time (s)")
plt.ylabel("flow (pl/s)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
plt.locator_params(axis='both', nbins=4)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_axial_outflow.png")

