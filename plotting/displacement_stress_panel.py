
import sys
sys.path.append('.')
from plotting.utils import load_results, mesh_name, sim_name, m3topl
 
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

results_closed = load_results("cardiac-closed", "arteriole")
results_open = load_results("cardiac-open", "arteriole")

flows = ['aqp_membrane_flow',
        'non_aqp_membrane_flow',
        'ecs_flow',
        'endfeet_neck_flow',
        #'inlet_flow',
        #'outlet_flow',
        #'membrane_volumetric_flow_rate'
        ]

area_names = ["aqp_membrane_id_area",
         "interf_id_area",
         "ecs_outer_id_area",
         "endfeet_neck_outer_id_area"]

flow_labels = ["AQ", "NAQ", "ECS", "EFN"]


fig = plt.figure(tight_layout=True, figsize=(8,6))
gs = gridspec.GridSpec(2, 2)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
ax4 = fig.add_subplot(gs[3])

# mean von mises stress
width = 0.4
D = ["ics", "pvs", "ecs"]

x = np.arange(3)
mean_mises_closed = np.array([results_closed[f"{d}_von_mises"].max() for d in D])
mean_mises_open = np.array([results_open[f"{d}_von_mises"].max() for d in D])
ax1.bar(x - width/2, mean_mises_open, width, label="open")
ax1.bar(x + width/2, mean_mises_closed, width, label="closed")
ax1.set_ylabel("stress (Pa)")
ax1.set_xticks(x, [d.upper() for d in D])

x = np.arange(3)
mean_mises_closed = np.array([results_closed[f"{d}_displacement"].max() for d in D])
mean_mises_open = np.array([results_open[f"{d}_displacement"].max() for d in D])
ax2.bar(x - width/2, mean_mises_open*1e9, width)
ax2.bar(x + width/2, mean_mises_closed*1e9, width)
ax2.set_ylabel("displacement (nm)")
ax2.set_xticks(x, [d.upper() for d in D])

plt.figlegend(loc='upper center',
              ncol=5, frameon=False)

sns.despine()
for i,ax in enumerate(fig.axes):
    ax.set_title("ABCDEFGHIJKLMN"[i], loc='left', fontweight="bold")
fig.tight_layout(rect=(0,0,1,0.95))
plt.savefig("disp_stress_panel.png")