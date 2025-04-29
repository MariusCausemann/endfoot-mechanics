
import sys 
sys.path.append('.')

from plotting.utils import load_results, mesh_name, sim_name, set_plotting_defaults, efcolor, pvscolor, ecscolor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
times = results["times"]
ecs_outer = results["ecs_outer_mean_pressure"]
astro_outer = results["astro_outer_mean_pressure"]
astro_interf = results["astro_interf_mean_pressure"]
ecs_interf = results["ecs_interf_mean_pressure"]

set_plotting_defaults()
plt.figure()
plt.plot(times, ecs_outer, label="ecs outer")
plt.plot(times, astro_outer, label="endfeet outer")
plt.plot(times, astro_interf, label="endfeet membrane")
plt.plot(times, ecs_interf, label="ecs membrane")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("pore pressure [Pa]")

if __name__ == '__main__':
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_surface_pore_pressure.png")

