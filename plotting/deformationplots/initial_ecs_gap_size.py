
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
counts = results["ecs_volume_counts"][0]
binpoints = results["ecs_volume_binpoints"]

peak_width = binpoints[np.argmax(counts)]*1e3
print(peak_width)

sns.set_context("talk")
plt.figure(dpi=dpi)
sns.histplot(x=binpoints*1000, weights=counts, kde=True, stat="percent",
             color=ecscolor, binrange=(0, 1000), bins=len(binpoints),
             edgecolor="white", kde_kws={"bw_adjust":.1})
plt.xlabel("ECS local width (nm)")
plt.ylabel("relative frequency (%)")
plt.xlim(0,500)
plt.annotate(f"{peak_width:.0f} nm", (peak_width, counts.max()), color="red", 
            textcoords="offset points", xytext=(4,4))
plt.axvline(peak_width, linestyle="--", lw=1.5, color="red")
#plt.ylim(0, 6)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_initial_ecs_gaps.png")