
import sys
sys.path.append('.')
from plotting.utils import load_results, set_plotting_defaults, efcolor, pvscolor, ecscolor, dpi
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

set_plotting_defaults()

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
initial_count = results["ecs_volume_counts"][0]
expand_count = results["ecs_volume_counts"][np.argmax(results["vessel_diam"])]
contract_count = results["ecs_volume_counts"][np.argmin(results["vessel_diam"])]
binpoints = results["ecs_volume_binpoints"]

plt.figure(dpi=dpi)
for counts, col,label in zip([contract_count, initial_count, expand_count],
                             ["navy","darkgreen", "darkred"],
                             ["contracted", "neutral", "expanded"]):
    sns.ecdfplot(x=binpoints*1000, weights=counts,stat="percent",
                color=col, alpha=0.5, label=label, lw=2, ls="-.")
plt.xlabel("ECS local width (nm)")
plt.ylabel("proportion (%)")
plt.xlim(0,800)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3,
           columnspacing=0.2, frameon=False)
#plt.ylim(0, 6)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_ecs_gaps_comp.png")