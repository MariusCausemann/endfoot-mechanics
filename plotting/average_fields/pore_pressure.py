
import sys 
sys.path.append('.')

from plotting.utils import load_results, mesh_name, sim_name, set_plotting_defaults, efcolor, pvscolor, ecscolor, occolor,dpi
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
times = results["times"]
ef_p = results["endfeet_pore_pressure"][1,:]
ecs_p = results["ecs_pore_pressure"][1,:]
pvs_p = results["pvs_pore_pressure"][1,:]
oc_p = results["oc_pore_pressure"][1,:]

set_plotting_defaults()
fig,ax = plt.subplots(dpi=dpi)
plt.plot(10*times, ef_p,color=efcolor, label="EF")
plt.plot(10*times, pvs_p, color=pvscolor, ls="dashdot", label="PVS")
plt.plot(10*times, ecs_p, color=ecscolor,label="ECS")
plt.plot(10*times, oc_p, color=occolor,label="OC", ls="--")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False)
plt.xlabel("cardiac cycles")
plt.ylabel("pore pressure (Pa)")
plt.locator_params(axis='both', nbins=5)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_pore_pressure.png")