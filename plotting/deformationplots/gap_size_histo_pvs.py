import sys
sys.path.append('.')
from plotting.utils import load_results, mesh_name, sim_name, gap_size_histo
 
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
gap_size_histo(results["pvs_volume_counts"], results["pvs_volume_binpoints"],
            results["times"]) 
if __name__ == '__main__':
    plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_pvs_gaps.png")

