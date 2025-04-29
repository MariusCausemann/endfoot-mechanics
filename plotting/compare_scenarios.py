
import sys
sys.path.append('.')
from plotting.utils import (load_results, mesh_name, sim_name,
                            m3topl, efcolor, ecscolor, pvscolor)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import pandas as pd
import matplotlib.gridspec as gridspec
sns.set_theme()
sns.set_style(style='ticks')

m32fl = 1e18
m2nm = 1e9
m2mum = 1e6

def add_percentage_change(ax, xvals, bs):
    b1, b2 = bs
    for x, bi1, bi2 in zip(xvals, b1, b2):
        reldiff = 100 * (bi2 - bi1)/bi1
        ax.annotate(f"{reldiff:.0f}%", 
                    (x,  max([bi1, bi2])), horizontalalignment="center",
                      verticalalignment="bottom", textcoords='offset pixels', xytext=(0, 2))


flows = [
        'endfeet_gap_flow',
        'endfeet_neck_flow',
        'aqp_membrane_flow',
        'non_aqp_membrane_flow',
        'ecs_flow',
        #"outlet_flow"
        #'inlet_flow',
        #'outlet_flow',
        #'membrane_volumetric_flow_rate'
        ]

area_names = [
        "endfeet_gap_id_area",
        "endfeet_neck_outer_id_area",
        "aqp_membrane_id_area",
         "interf_id_area",
         "ecs_outer_id_area",
         #"pvs_outlet_id_area"
         ]

flow_labels = ["EFG","AP","AdM", "AbM", "ECS"] # 
show_signs = False

colordict = {"WT":"purple", "euler":"cornflowerblue", 
            "closed":"purple", "high resistance":"purple","maximal resistance":"brown", 
          "open":"teal","low resistance":"teal", "minimal resistance":"goldenrod",
           "AQP4-KO":"darkgoldenrod",
          "EF-KO":"lightsalmon", "lagrange":"gold",
          "WT-VLF":"cornflowerblue","AQP4-KO-VLF":"black",
          "highPerm" : "slategrey", "stiff":"deeppink", "closedouter":"blueviolet",
          "standard":"black", "refined":"red"
          } 
colordict.update({i:c for i,c in zip(["baseline","x2","x4","x6","x8"], sns.color_palette("mako"))})
print(colordict)
def get_abs_max(ar):
    m = max(ar, key=abs)
    return abs(m), "-" if m > 0 else "+"


def plot_panel(results, scenario_names, filename, layout="3x2"):
    size = (2,3)
    #size = (1,6)
    #size = (3,2)
    fig = plt.figure(tight_layout=True, figsize=(size[1]*3, size[0]*3))# if layout=="3x2" else (16,8))
    gs = gridspec.GridSpec(*size)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    ax6 = fig.add_subplot(gs[5])

    colors = [colordict.get(n, None) for n in scenario_names]
    # max flow rates
    nr = len(results)
    width = {2:0.4, 3:0.4, 4:0.3, 5:0.35}[nr]

    x = np.arange(len(flows))
    max_flows = [np.zeros(len(flows)) for i in results]
    signs = [[None]*len(flows) for i in results]
    for j,r in enumerate(results):
        for i, q in enumerate(flows):
            max_flows[j][i], signs[j][i] = get_abs_max(r[q])

    for i,(mf,c) in enumerate(zip(max_flows, colors)):
        b1 = ax1.bar(x - width/2 + i*width*2 / nr, mf*m32fl, width, color=c)

    if show_signs:
        for i, (bar1, bar2) in enumerate(zip(b1, b2)):
            ax1.text(bar1.get_x() + width/2, bar1.get_height(),
                    signs_1[i], ha='center', va='bottom')
            ax1.text(bar2.get_x() + width/2, bar2.get_height(),
                    signs_2[i], ha='center', va='bottom')
        
    ax1.set_yscale("log")
    ax1.set_ylabel("flow ($\mu m^3$/s)")
    ax1.set_xticks(x, flow_labels)
    if nr==2: add_percentage_change(ax1, x, [mf*m32fl for mf in max_flows])

    # max flux rates
    areas = np.array([results[0][an] for an in area_names])
    max_flux = [mf / areas for mf in max_flows]

    for i,(mf,c) in enumerate(zip(max_flux, colors)):
        b1 = ax2.bar(x - width/2 + i*width*2 / nr, mf*m2nm, width, color=c)

    if show_signs:
        for i, (bar1, bar2) in enumerate(zip(b1, b2)):
            ax2.text(bar1.get_x() + width/2, bar1.get_height(),
                    signs_1[i], ha='center', va='bottom')
            ax2.text(bar2.get_x() + width/2, bar2.get_height(),
                    signs_2[i], ha='center', va='bottom')
    ax2.set_ylabel("flux (nm/s)")
    ax2.set_xticks(x, flow_labels)
    ax2.set_yscale("log")
    if nr==2: add_percentage_change(ax2, x, [mf*m2nm for mf in max_flux])


    # max mean pressure differences

    x = np.arange(2)
    ecs_outer_mean_p = [r["ecs_outer_mean_pressure"] for r in results]

    aqp_dp= [abs(r["astro_interf_mean_pressure"] - 
                        r["ecs_interf_mean_pressure"]).max() for r in results]
    layer_dp = [abs(r["ecs_outer_mean_pressure"] - 
                        r["ecs_interf_mean_pressure"]).max() for r in results]

    for i, (aqp, ldp, c) in enumerate(zip(aqp_dp, layer_dp, colors)):
        ax3.bar(x - width/2 + i*width*2 / nr, [aqp, ldp], width, color=c)

    ax3.set_ylabel("pressure difference (Pa)")
    ax3.set_xticks(x, ["AQP \n membrane", "endfoot \n sheath"])

    if nr==2: add_percentage_change(ax3, x, [(aqp, ldp) for aqp, ldp in zip(aqp_dp, layer_dp)])

    # mean flow velocities

    D = ["ef", "oc", "pvs", "ecs"]

    x = np.arange(4)
    mean_flow_vel = [np.array([r[f"mean_{d}_flow"].max() for d in D]) for r in results]

    for i, (mf, c) in enumerate(zip(mean_flow_vel, colors)):
        ax4.bar(x - width/2 + i*width*2 / nr, mf*m2mum, width, color=c)

    ax4.set_ylabel("velocity ($\mu$m/s)")
    ax4.set_xticks(x, [d.upper() for d in D])
    if nr==2: add_percentage_change(ax4, x, [mf*m2mum for mf in mean_flow_vel])

    mean_mises = [np.array([r[f"{d}_von_mises"].max() for d in D]) for r in results]
    for i, (mm, c) in enumerate(zip(mean_mises, colors)):
        ax5.bar(x - width/2 + i*width*2 / nr, mm, width, color=c)
    ax5.set_ylabel("stress (Pa)")
    ax5.set_xticks(x, [d.upper() for d in D])
    if nr==2: add_percentage_change(ax5, x, [mm for mm in mean_mises])

    mean_disp = [np.array([r[f"{d}_displacement"].max() for d in D]) for r in results]
    for i, (md, c) in enumerate(zip(mean_disp, colors)):
        ax6.bar(x - width/2 + i*width*2 / nr, md*1e9, width, color=c)
    ax6.set_ylabel("displacement (nm)")
    ax6.set_xticks(x, [d.upper() for d in D])
    if nr==2: add_percentage_change(ax6, x, [md*1e9 for md in mean_disp])

    # displacement over time

    from deformationplots.diameter_change import get_diameter
    times = results[0]["times"]

    diams = [get_diameter(r) for r in results] #lumen_diam, pvs_diam_1, ef_diam_1
    pvs_widths = [(pvs_diam - lumen_diam) / 2 for (lumen_diam, pvs_diam, ef_diam) in diams]


    """
    gs7 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[6:9])

    ax71 = fig.add_subplot(gs7[0])
    ax72 = fig.add_subplot(gs7[1])
    ax73 = fig.add_subplot(gs7[2])

    marker = ["-", "--",":", "-,"]
    for n, m, pvs_w, (lumen_diam, pvs_diam, ef_diam) in zip(scenario_names, marker, pvs_widths, diams):
        ax71.plot(times, 1e3*(ef_diam - ef_diam[0]), m, color=efcolor, label=f"$\Delta$ EF ({n})")
        ax72.plot(times, 1e3*(pvs_w - pvs_w[0]), m, color=ecscolor, label=f"$\Delta$ PVS ({n})")
    ax73.plot(times, 1e3*(lumen_diam - lumen_diam[0]), color="crimson", label="$\Delta$ lumen")

    ax73.set_xlabel("time (s)")
    #ax7.legend(frameon=False, loc="center left", ncol=1,
    #           bbox_to_anchor=(0, 0.5), labelspacing=4)
    #ax71.legend(frameon=False, loc="upper center", bbox_to_anchor=(.1, 1.4), ncols=2)
    ax71.legend(frameon=False, ncols=min([nr,3]), loc='upper left', bbox_to_anchor=(0., 1.6))
    ax72.legend(frameon=False, ncols=min([nr,3]), loc='upper left', bbox_to_anchor=(0., 1.6))
    ax73.legend(frameon=False, ncols=min([nr,3]), loc='upper left', bbox_to_anchor=(0., 1.6))

    sns.despine()
    ax71.set_ylabel("$\Delta$ endfeet")
    ax72.set_ylabel("$\Delta$ PVS")

    ax71.spines['left'].set_visible(False)
    ax72.spines['left'].set_visible(False)
    ax73.spines['left'].set_visible(False)
    ax73.set_yticks([])
    ax73.vlines(-0.01, ymin=-50, ymax=50, color="crimson", lw=5)
    ax72.vlines(-0.01, ymin=-5, ymax=5, color=ecscolor,lw=5)
    ax71.vlines(-0.01, ymin=-50, ymax=50, color="teal",lw=5)
    #ax73.text(-0.02, 0, "80 nm", rotation="vertical", color="red")
    ax71.set_ylabel("100 nm", color=efcolor)
    ax72.set_ylabel("10 nm", color=ecscolor)
    ax73.set_ylabel("100 nm", color="crimson")
    ax71.set_xlim(left=-0.01)
    ax72.set_xlim(left=-0.01)
    ax73.set_xlim(left=-0.01)

    ax71.spines['left'].set_visible(False)
    ax71.set_yticks([])
    ax71.spines['bottom'].set_visible(False)
    ax71.set_xticks([])

    ax72.spines['left'].set_visible(False)
    ax72.set_yticks([])
    ax72.spines['bottom'].set_visible(False)
    ax72.set_xticks([])
    """

    sns.despine()
    plt.figlegend(scenario_names, loc='upper center', #fontsize="large",
                ncol=min(nr, size[1]), frameon=False)

    fig.tight_layout(rect=(0,0,1,0.95))
    #fig.subplots_adjust(wspace=0.35)
    axs = [ax1, ax2,ax3,ax4,ax5,ax6]
    if size[0] > 1:
        for i in range(size[0]):
            fig.align_ylabels(axs[i::size[1]])
    plt.savefig(filename, transparent=True, dpi=500)

if __name__ == '__main__':

    n = (len(sys.argv) -1) / 3
    print(sys.argv)
    sim_names = sys.argv[1::3]
    mesh_names = sys.argv[2::3]
    scenario_names = sys.argv[3::3]
    print(sim_names)
    print(mesh_names)
    print(scenario_names)

    folder = "_".join([f"{sn}_{mn}" for sn, mn in zip(mesh_names, sim_names)])
    print(folder) 
    filename = (f"results/comparisons/{folder}/" + 
               f"{'_'.join(scenario_names)}.png")
    results = [load_results(sn, mn) for sn, mn in zip(mesh_names, sim_names)]
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    plot_panel(results, [n.replace("+", " ") for n in scenario_names], filename)




