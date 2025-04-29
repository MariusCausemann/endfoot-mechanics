
import sys 
sys.path.append('.')

from plotting.utils import load_results, mesh_name, sim_name, set_plotting_defaults, efcolor, pvscolor, ecscolor
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
times = results["times"]
u_y_mean = results["mean_y_velocity"]

set_plotting_defaults()
plt.figure()
plt.plot(times, u_y_mean * 1e6, label="pvs mean axial velocity")
plt.axhline(u_y_mean.mean()*1e6, label="temporal mean")
plt.legend()
plt.xlabel("time [s]")
plt.ylabel("$\overline{u}_y$  [$\mu$m/s]")

if __name__ == '__main__':
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_mean_pvs_velocity.png")
