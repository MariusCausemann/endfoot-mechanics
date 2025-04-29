
import sys
sys.path.append('.')
from plotting.utils import load_results, m3tofl, set_plotting_defaults, efcolor, pvscolor, ecscolor, dpi
import matplotlib.pyplot as plt
set_plotting_defaults() 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
times = results["times"]
ecs_outflow = -results["ecs_flow"]
ef_outflow = -results["endfeet_neck_flow"]

fig,ax = plt.subplots(dpi=dpi)
plt.plot(10*times, ef_outflow*m3tofl,efcolor, label="astrocyte process")
plt.plot(10*times, ecs_outflow*m3tofl, ecscolor, label="ECS")
plt.xlabel("cardiac cycles")
plt.ylabel("inflow ($\mu m^3$/s)")
plt.locator_params(axis='both', nbins=5)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_radial_outflow.png")

