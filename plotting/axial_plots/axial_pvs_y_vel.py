

import sys
sys.path.append('.')
from plotting.utils import load_results, mesh_name, sim_name, m3topl
 
import matplotlib.pyplot as plt
import numpy as np
import yaml


if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
with open(f"config_files/{sim_name}.yml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
T = config["T"]
f = config["f"]
times = results["times"]
y_coords = results["axial_y_coordinates"]
ax_pvs_p = results["pvs_axial_y_velocity"]
plot_times = np.linspace(T - 1/f, T, 6)
plot_idx = [np.argmin(abs(times - pt)) for pt in plot_times]
plt.figure()
for i in plot_idx:
    plt.plot(y_coords, ax_pvs_p[:,i]*1e6, label=f"t={np.round(times[i],2):.2f}s")
plt.xlabel("y [\u03BCm]")
plt.ylabel("velocity [\u03BCm/s]")
plt.legend()
if __name__ == '__main__':
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_pvs_axial_y_velocity.png")

