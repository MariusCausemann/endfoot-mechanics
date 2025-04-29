import sys
sys.path.append('.')

from plotting.utils import (load_results, set_plotting_defaults,
                             efcolor, pvscolor, ecscolor,occolor, dpi,draw_brace)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
set_plotting_defaults()
from cmap import Colormap

efcmap = Colormap("colorbrewer:brbg_6")

plt.rcParams['svg.fonttype'] = 'none'

if __name__ == '__main__':
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]

results = load_results(mesh_name, sim_name)
astro_keys = [k for k in results.keys() if k.startswith("astrocyte_")]
times = results["times"]
ef_vol = results["ef_volume"] 
pvs_vol = results["pvs_volume"] 
ecs_vol = results["ecs_volume"] 
oc_vol = results["oc_volume"] 
print(ef_vol[0])
print(ecs_vol[0])
print(pvs_vol[0])
alpha = ecs_vol[0] / (ecs_vol[0] + ef_vol[0] + oc_vol[0])

sns.set_context("talk")
print(f"alpha = {100*alpha:.1f}%")
labels = ["PVS","ECS"]
vols = [pvs_vol[0], ecs_vol[0]]
astrovols = [results[ak][0] for ak in astro_keys if results[ak][0] > 0]
allvols = vols + astrovols + [oc_vol[0]]
fig,ax = plt.subplots(dpi=dpi)
plt.bar(range(len(allvols)), allvols, 
        tick_label=labels + [f"EF{i + 1}" for i in range(len(astrovols))] + ["OC"],
        color=[pvscolor, ecscolor] + list(efcmap.iter_colors(len(astrovols))) + [occolor])
for i, vol in enumerate(allvols):
    plt.annotate(f"{100*vol/sum(allvols):.0f}%", xy=(i, vol+ 100),
                  horizontalalignment="center",
                    verticalalignment="bottom")
    
draw_brace(plt.gca(), (1.7, 1.5+ len(astrovols)), 1000, 
           f"EF: {100*sum(astrovols)/sum(allvols):.0f}%", color="black")
    
plt.ylabel(f"volume (μm³)")
plt.xlabel("compartment",labelpad=3)
plt.xlim(-0.5, 2.8 + len(astrovols))
plt.ylim(0.5, 2200)
plt.xticks(rotation=30)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_initial_vol.png")

rel_ef_vol_change = (ef_vol - ef_vol[0]) / ef_vol[0]
rel_pvs_vol_change = (pvs_vol - pvs_vol[0]) / pvs_vol[0]
rel_ecs_vol_change = (ecs_vol - ecs_vol[0]) / ecs_vol[0]
rel_oc_vol_change = (oc_vol - oc_vol[0]) / oc_vol[0]

set_plotting_defaults()
fig,ax = plt.subplots(dpi=dpi)
plt.plot(10*times, 100*rel_ef_vol_change, color=efcolor, label="EF")
plt.plot(10*times, 100*rel_pvs_vol_change, color=pvscolor, ls="dashdot", label="PVS")
plt.plot(10*times, 100*rel_ecs_vol_change, color=ecscolor, label="ECS")
plt.plot(10*times, 100*rel_oc_vol_change, color=occolor, label="OC", ls="--")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False)
plt.xlabel("cardiac cycles")
plt.ylabel("relative volume change (%)")
plt.locator_params(axis='both', nbins=5)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_rel_volume_change.png")


fig,ax = plt.subplots(dpi=dpi)
plt.plot(10*times, ef_vol - ef_vol[0], color=efcolor, label="EF")
plt.plot(10*times, pvs_vol - pvs_vol[0], color=pvscolor, ls="dashdot", label="PVS")
plt.plot(10*times, ecs_vol - ecs_vol[0], color=ecscolor, label="ECS")
plt.plot(10*times, oc_vol - oc_vol[0], color=occolor, label="OC", ls="--")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False)
plt.xlabel("cardiac cycles")
plt.ylabel("volume change (μm³)")
plt.locator_params(axis='both', nbins=5)    
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_volume_change.png")

plt.figure(dpi=dpi)
for ak in astro_keys:
    vol = results[ak]
    initial_vol = vol[0]
    plt.plot(10*times, 100* (vol - initial_vol)/initial_vol , label=ak)
plt.xlabel("cardiac cycles")
plt.ylabel("relative volume change (%)")
plt.locator_params(axis='both', nbins=5)
ax.yaxis.set_label_coords(0.08, 0.5, transform=fig.transFigure)
plt.savefig(f"results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_astrocyte_volume_change.png")



