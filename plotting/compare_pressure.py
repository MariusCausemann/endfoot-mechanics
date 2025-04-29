import sys 
sys.path.append('.')
import seaborn as sns
from plotting.utils import load_results, mesh_name, sim_name, set_plotting_defaults, efcolor, pvscolor, ecscolor, occolor, dpi
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sim_names = sys.argv[1::2]
    mesh_names = sys.argv[2::2]
ns = len(sim_names)
print(sim_names)
results = []
for mesh_name, sim_name in zip(mesh_names, sim_names):
    results.append(load_results(mesh_name, sim_name))

times = results[0]["times"]
labels =  ["baseline", "x2", "x4", "x6", "x8"]
linestyles = ["-", "--", "-", "--", "-"]
set_plotting_defaults()
sns.set_context("talk")
colors = sns.color_palette("mako")
handlelength = 1.2
columnspacing = 0.6

fig,ax = plt.subplots(dpi=dpi)
for r,l, c,ls in zip(results, labels, colors, linestyles):
    pvs_p = r["pvs_pore_pressure"][1,:]
    plt.plot(10*times, pvs_p, color=c, label=l, ls=ls)
plt.xlabel("cardiac cycles")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
 ncol=ns, frameon=False, handlelength=handlelength, columnspacing=columnspacing)
plt.ylabel("PVS pressure (Pa)")
ax.yaxis.set_label_coords(0.12, 0.55, transform=fig.transFigure)
plt.savefig("pressure.png")

fig,ax = plt.subplots(dpi=dpi)
for r,l, c,ls in zip(results, labels, colors, linestyles):
    print(c)
    ef_flow = r["endfeet_gap_flow"]
    plt.plot(10*times, ef_flow*1e18, color=c, label=l, ls=ls)
plt.xlabel("cardiac cycles")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=ns,
 frameon=False, handlelength=handlelength, columnspacing=columnspacing)
plt.ylabel("EF gap flow (μm³/s)")
ax.yaxis.set_label_coords(0.12, 0.55, transform=fig.transFigure)
plt.savefig("ef_gap_flow.png")

fig,ax = plt.subplots(dpi=dpi)
for r,l, c, ls in zip(results, labels, colors, linestyles):
    print(c)
    pvs_vol = r["pvs_volume"]
    plt.plot(10*times, pvs_vol - pvs_vol[0], color=c, label=l, ls=ls)
plt.xlabel("cardiac cycles")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
     ncol=ns, frameon=False, handlelength=handlelength, columnspacing=columnspacing)
plt.ylabel("PVS volume change (μm³)")
ax.yaxis.set_label_coords(0.12, 0.55, transform=fig.transFigure)
plt.savefig("volume_change.png")

