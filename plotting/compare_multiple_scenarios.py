
import sys
sys.path.append('.')
from plotting.utils import load_results, mesh_name, sim_name, m3topl
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yaml
import pandas as pd
import matplotlib.gridspec as gridspec
import argparse

sns.set_theme()
sns.set_style(style='ticks')

m32fl = 1e18
m2nm = 1e9
m2mum = 1e6

flows = ['aqp_membrane_flow',
        'non_aqp_membrane_flow',
        'endfeet_gap_flow',
        'ecs_flow',
        'endfeet_neck_flow',
        #"outlet_flow"
        #'inlet_flow',
        #'outlet_flow',
        #'membrane_volumetric_flow_rate'
        ]

area_names = ["aqp_membrane_id_area",
         "interf_id_area",
         "endfeet_gap_id_area",
         "ecs_outer_id_area",
         "endfeet_neck_outer_id_area",
         #"pvs_outlet_id_area"
         ]

flow_labels = ["AQD", "AQS","EFG", "ECS", "EFN",] # 
show_signs = False

colors = {"WT":"cornflowerblue", "euler":"cornflowerblue", "closed":"cornflowerblue",
          "open":"gold", "AQP4-KO":"teal",
          "EF-KO":"lightsalmon", "lagrange":"gold",
          "WT-VLF":"cornflowerblue","AQP4-KO-VLF":efcolor,
          "highPerm" : "slategrey", "stiff":"deeppink", "closedouter":"blueviolet"}

def get_abs_max(ar):
    m = max(ar, key=abs)
    return abs(m), "-" if m > 0 else "+"


def plot_panel(results1, results2, scenario_names, filename):
    size = (3,2)
    fig = plt.figure(tight_layout=True, figsize=np.array(size)*3)
    gs = gridspec.GridSpec(*size)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])
    ax5 = fig.add_subplot(gs[4])
    ax6 = fig.add_subplot(gs[5])
    scn1, scn2 = scenario_names
    c1, c2 = colors[scn1], colors[scn2]

    # max flow rates
    width = 0.4
    x = np.arange(len(flows))
    max_flow_1, max_flow_2 = {"sn":scenario_names[0]}, {"sn":scenario_names[1]}
    signs_1, signs_2 = [None]*len(flows), [None]*len(flows)
    for i, q in enumerate(flows):
        max_flow_1[flow_labels[i]] = [get_abs_max(r[q])[0]*m32fl for r in results1]
        max_flow_2[flow_labels[i]] = [get_abs_max(r[q])[0]*m32fl for r in results2]

    df = pd.concat([pd.DataFrame(max_flow_1), pd.DataFrame(max_flow_2)])
    mdf = pd.melt(df, id_vars=["sn"], var_name="flow", value_name="flow ($\mu m^3$/s)")
    plot = sns.stripplot(mdf, x="flow", y="flow ($\mu m^3$/s)", hue="sn", dodge=True, size=7, linewidth=1,
                         ax=ax1)
    plot.set(xlabel=None)
    plot.legend_.remove()

    # max flux rates
    areas = np.array([results1[0][an] for an in area_names])
    for col, area in zip(flow_labels, areas):
        df[col] /= area * m2nm

    mdf = pd.melt(df, id_vars=["sn"], var_name="flow", value_name="flux (nm/s)")
    plot = sns.stripplot(mdf, x="flow", y="flux (nm/s)", hue="sn", dodge=True, size=7, linewidth=1,
                         ax=ax2)
    plot.set(xlabel=None)
    plot.legend_.remove()

    # mean pressure difference
    dp1, dp2 = {"sn":scenario_names[0]}, {"sn":scenario_names[1]}
    dp1["AQP \n membrane"] = [abs(r["astro_interf_mean_pressure"] - 
                                    r["ecs_interf_mean_pressure"]).max() for r in results1]
    dp2["AQP \n membrane"] = [abs(r["astro_interf_mean_pressure"] - 
                                    r["ecs_interf_mean_pressure"]).max() for r in results2]

    dp1["endfoot \n sheath"] = [abs(r["ecs_outer_mean_pressure"] - 
                                    r["ecs_interf_mean_pressure"]).max() for r in results1]
    dp2["endfoot \n sheath"] = [abs(r["ecs_outer_mean_pressure"] - 
                                    r["ecs_interf_mean_pressure"]).max() for r in results2]

    df = pd.concat([pd.DataFrame(dp1), pd.DataFrame(dp2)])
    mdf = pd.melt(df, id_vars=["sn"], var_name="dp", value_name="pressure difference (Pa)")
    plot = sns.stripplot(mdf, x="dp", y="pressure difference (Pa)", hue="sn", dodge=True,
                         size=7, linewidth=1, ax=ax3)
    plot.set(xlabel=None)
    plot.legend_.remove()


    # mean flow velocities

    D = ["ics", "pvs", "ecs"]
    mean_flows_1, mean_flows_2 = {"sn":scenario_names[0]}, {"sn":scenario_names[1]}

    for d in D:
        mean_flows_1[d.upper()] = [np.array(r[f"mean_{d}_flow"]).max()*m2mum for r in results1] 
        mean_flows_2[d.upper()] = [np.array(r[f"mean_{d}_flow"]).max()*m2mum for r in results2]
    df = pd.concat([pd.DataFrame(mean_flows_1), pd.DataFrame(mean_flows_2)])
    mdf = pd.melt(df, id_vars=["sn"], var_name="vel", value_name="velocity ($\mu$m/s)")
    plot = sns.stripplot(mdf, x="vel", y="velocity ($\mu$m/s)", hue="sn", dodge=True,
                         size=7, linewidth=1, ax=ax4)
    plot.set(xlabel=None)
    plot.legend_.remove()

    mean_mises_1, mean_mises_2 = {"sn":scenario_names[0]}, {"sn":scenario_names[1]}

    for d in D:
        mean_mises_1[d.upper()] = [np.array(r[f"{d}_von_mises"]).max() for r in results1] 
        mean_mises_2[d.upper()] = [np.array(r[f"{d}_von_mises"]).max() for r in results2]
    df = pd.concat([pd.DataFrame(mean_mises_1), pd.DataFrame(mean_mises_2)])
    mdf = pd.melt(df, id_vars=["sn"], var_name="von_mises", value_name="stress (Pa)")
    plot = sns.stripplot(mdf, x="von_mises", y="stress (Pa)", hue="sn", dodge=True,
                         size=7, linewidth=1, ax=ax5)
    plot.set(xlabel=None)
    plot.legend_.remove()

    mean_disp_1, mean_disp_2 = {"sn":scenario_names[0]}, {"sn":scenario_names[1]}

    for d in D:
        mean_disp_1[d.upper()] = [np.array(r[f"{d}_displacement"]).max()*m2nm for r in results1] 
        mean_disp_2[d.upper()] = [np.array(r[f"{d}_displacement"]).max()*m2nm for r in results2]
    df = pd.concat([pd.DataFrame(mean_disp_1), pd.DataFrame(mean_disp_2)])
    mdf = pd.melt(df, id_vars=["sn"], var_name="disp", value_name="displacement (nm)")
    plot = sns.stripplot(mdf, x="disp", y="displacement (nm)", hue="sn", dodge=True,
                         size=7, linewidth=1, ax=ax6)
    plot.set(xlabel=None)
    plot.legend_.remove()


    sns.despine()
    plt.figlegend(scenario_names, loc='upper center',
                ncol=5, frameon=False)

    fig.tight_layout(rect=(0,0,1,0.95))
    plt.savefig(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s1', '--scenarios1', nargs='+', default=[])
    parser.add_argument('-s2', '--scenarios2', nargs='+', default=[])
    parser.add_argument('-n1', '--name1', type=str)
    parser.add_argument('-n2', '--name2', type=str)
    parser.add_argument('-f', '--filename', type=str)

    args = vars(parser.parse_args())

    results1 = [load_results(*name.split("_")) for name in args["scenarios1"]]
    results2 = [load_results(*name.split("_")) for name in args["scenarios2"]]
    filename = args["filename"]
    scenario_names = [args["name1"], args["name2"]]
    Path(filename).parent.mkdir(exist_ok=True, parents=True)
    plot_panel(results1, results2, scenario_names, filename)




