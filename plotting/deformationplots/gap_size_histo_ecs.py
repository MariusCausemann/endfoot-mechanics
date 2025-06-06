import sys
sys.path.append('.')
from plotting.utils import load_results, mesh_name, sim_name, gap_size_histo
 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
gap_size_histo(results["ecs_volume_counts"], results["ecs_volume_binpoints"],
            results["times"]) 
plt.ylim((0, 200))
if __name__ == '__main__':
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_ecs_gaps.png")

