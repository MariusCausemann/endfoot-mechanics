
import sys 
sys.path.append('.')
sys.path.append('..')


from plotting.utils import load_results, set_plotting_defaults, efcolor, pvscolor, ecscolor,occolor, dpi
import matplotlib.pyplot as plt
set_plotting_defaults()

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    
results = load_results(mesh_name, sim_name)
times = results["times"]
ef_vm = results["endfeet_von_mises"][1]
ecs_vm = results["ecs_von_mises"]
pvs_vm = results["pvs_von_mises"]
oc_vm = results["oc_von_mises"]

fig,ax = plt.subplots(dpi=dpi)
plt.plot(10*times, ef_vm, color=efcolor, label="EF")
plt.plot(10*times, pvs_vm, color=pvscolor, ls="dashdot", label="PVS")
plt.plot(10*times, ecs_vm, color=ecscolor, label="ECS")
plt.plot(10*times, oc_vm, color=occolor, label="OC", ls="--")
plt.xlabel("cardiac cycles")
plt.ylabel("von mises stress (Pa)")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False)
plt.locator_params(axis='both', nbins=5)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_von_mises.png")
