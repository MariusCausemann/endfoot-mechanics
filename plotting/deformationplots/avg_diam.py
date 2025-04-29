import sys
sys.path.append('.')
from plotting.utils import load_results, mesh_name, sim_name, line_data_over_time, ecscolor
 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

"""
results = load_results(mesh_name, sim_name)
num_steps = results["sim_config"]["num_steps"]
times = results["times"]
cmap = matplotlib.colors.ListedColormap(['lightgreen','grey','green', ecscolor,'crimson'])


avg_line_data = np.zeros((1000, num_steps))
r_max = np.ceil(results["ef_diam"].max() / 2)
x,y = np.meshgrid(range(num_steps), np.linspace(- r_max, r_max, 1000))
avg_line_data[abs(y) < results["ef_diam"] / 2] = 2
avg_line_data[abs(y) < results["pvs_diam"] / 2] = 3
avg_line_data[abs(y) < results["vessel_diam"] / 2] = 4
cmap = matplotlib.colors.ListedColormap(['lightgreen','grey','green', ecscolor,'crimson'])

line_data_over_time(avg_line_data.T, times, y[:,0], cmap)

#plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_avg_diam.png")
"""
