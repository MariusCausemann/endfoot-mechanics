import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import sys
import numpy as np
import k3d
from matplotlib.colors import LinearSegmentedColormap
import yaml

if __name__ == "__main__":
    sim_name = sys.argv[1]
    mesh_name = sys.argv[2]
    plot_name = sys.argv[3]
    if len(sys.argv) ==5:
        color = sys.argv[4]
    else:
        color = None

    sim_dir = f"results/{mesh_name}_{sim_name}"
    
    with open(f"{sim_dir}/{plot_name}/{plot_name}.yml") as f:
        plot_conf = yaml.load(f, Loader=yaml.FullLoader)
    with open(f"config_files/{sim_name}.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    color_map = plot_conf["cmap"]
    l = np.array(color_map).reshape(-1,4)[:,1:4]
    cmap = LinearSegmentedColormap.from_list("test",l.tolist())
    T = config["T"]
    f = config["f"]
    times = np.linspace(0, T, config["num_steps"])
    plot_times = np.linspace(T - 1/f, T, 6)
    plot_idx = [np.argmin(abs(times - pt)) for pt in plot_times]

    f = plt.figure()
    ax = f.add_subplot(1,1,1)
    im = ax.imshow(np.linspace(*plot_conf["scalar_range"], 100).reshape(10,10), cmap=cmap)
    
    fig = plt.figure(figsize=(12., 9.), frameon=True)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, int(np.ceil(len(plot_times)/2))),  # creates 2x2 grid of axes
                    axes_pad=0.0,  # pad between axes in inch.
                    cbar_mode="single", cbar_location="right", cbar_size="3%",cbar_pad=0.1,)


    for ti, ax, i in zip(plot_idx, grid, range(len(plot_times))):
        if color is not None:
            fn = f"{sim_dir}/{plot_name}_{color}/{plot_name}_{color}_{times[ti]:.3f}.png"
            textcolor = "black"
        else:
            fn = f"{sim_dir}/{plot_name}/{plot_name}_{times[ti]:.3f}.png"
            textcolor = "white"
        img = plt.imread(fn)
        ax.axis('off')
        ax.imshow(img[10:-10, 250:1000,:])
        ax.text(0.2, 0.9, f"{plot_times[i]:.2f} s", c=textcolor, size=12, transform=ax.transAxes)

    grid[0].cax.colorbar(im)
    cax = grid.cbar_axes[0]
    cax.axis[cax.orientation].set_label(plot_conf["cbar_label"])

    if color:
        plt.savefig(f"{sim_dir}/{plot_name}_grid_{color}.png", dpi=300, bbox_inches="tight",)
    else:
        plt.savefig(f"{sim_dir}/{plot_name}_grid.png", dpi=300, bbox_inches="tight",)

