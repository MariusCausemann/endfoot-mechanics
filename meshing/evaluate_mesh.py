import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import yaml

#pv.start_xvfb()

def generate_html(mesh_name):

    mesh = pv.read(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    bm = pv.read(f"meshes/{mesh_name}/{mesh_name}_facets.xdmf")
    vessel_wall = bm.extract_cells(np.isin(bm.cell_data["boundaries"], 4 ) ).extract_surface()
    vessel_wall.scale(1e6, inplace=True)
    mesh.scale(1e6, inplace=True)

    cells = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], [2,3,4,5,6,7,8] ))
    ecs = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], [1] ))

    pl = pv.Plotter(shape=(1,1), polygon_smoothing=True, window_size=(1200,1200))
    pl.add_mesh(cells, scalars="subdomains", cmap="tab20c")
    pl.add_mesh(vessel_wall, color="red", pbr=True, opacity=0.7)
    pl.background_color = "lightgrey"
    pl.export_html(f"meshes/{mesh_name}/{mesh_name}.html", backend="pythreejs")  


def evaluate_gap_sizes(mesh, domain_ids, object_ids):
    ecs = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], domain_ids ))
    ecs.clear_data()
    distances = []
    object_ids = list(set(object_ids) & set(mesh["subdomains"]))
    for i in object_ids:
        astro = mesh.extract_cells(np.isin(mesh.cell_data["subdomains"], [i] ))
        dist = ecs.compute_implicit_distance(astro.extract_surface())
        distances.append(abs(dist.point_data["implicit_distance"]))
        
    dist = ecs.compute_implicit_distance(mesh.extract_surface())
    distances.append(abs(dist.point_data["implicit_distance"]))

    distances = np.array(distances).T
    distances.sort()
    min_dist = distances[:,0] + distances[:,1]
    ecs["gap_size"] = min_dist
    ecs = ecs.point_data_to_cell_data()
    ecs = ecs.compute_cell_sizes()
    return ecs.cell_data["gap_size"], abs(ecs.cell_data["Volume"])

def plot_ecs_gap_size(mesh_name):
    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    mesh = pv.read(f"meshes/{mesh_name}/{mesh_name}.xdmf")
    mesh.scale(1e6)
    ecs = mesh.extract_cells(np.isin(mesh["subdomains"], mesh_config["fluid_comp_ids"]["ecs"]))
    object_ids = mesh_config["porous_cell_ids"]["astrocyte"] + mesh_config["porous_cell_ids"]["outer_cell"]
    min_dist, vol = evaluate_gap_sizes(mesh, mesh_config["fluid_comp_ids"]["ecs"],
                                  object_ids)


    n, bins, patches = plt.hist(min_dist*1000, 500, density=True, weights=vol)
    plt.xlim(-10, 300)
    plt.xlabel("gap size [nm]")
    plt.ylabel("ECS volume share")
    plt.grid()
    plt.savefig(f"meshes/{mesh_name}/{mesh_name}_gap_sizes.png")

def compute_subdomain_volume(mesh, subdomain_scalar="subdomains"):
    volumes = dict()
    for subd in np.unique(mesh.cell_data["subdomains"]):
        dom = mesh.extract_cells(mesh.cell_data["subdomains"] == subd)
        volumes[int(subd)] = float(dom.volume)
    return volumes

def get_id_to_type_map(mesh_config):
    cell_ids = mesh_config["porous_cell_ids"]
    fluid_ids = mesh_config["fluid_comp_ids"]
    id_dict = dict(**cell_ids, **fluid_ids)
    id_to_type_map = {}
    astro_counter = 0
    for dom_type, ids in id_dict.items():
        for id in ids:
            if dom_type=="astrocyte":
                astro_counter += 1
                id_to_type_map[id] = dom_type + f" {astro_counter}"
            else:
                id_to_type_map[id] = dom_type
    return id_to_type_map

        

def compute_volumes(mesh_name):

    with open(f"config_files/{mesh_name}.yml") as conf_file:
        mesh_config = yaml.load(conf_file, Loader=yaml.FullLoader)

    mesh = pv.read(f"meshes/{mesh_name}/{mesh_name}.xdmf")

    volumes = compute_subdomain_volume(mesh)
    id_to_type_map = get_id_to_type_map(mesh_config)
    volumes = {id_to_type_map[k]:v for k,v in volumes.items()}

    with open(f"meshes/{mesh_name}/{mesh_name}_volumes.yml", 'w') as outfile:
        yaml.dump(volumes, outfile, default_flow_style=False)

    plt.figure()
    plt.bar(*zip(*volumes.items()))
    plt.xticks(rotation=-20)
    plt.ylabel("Volume [pl]")
    plt.grid()
    plt.savefig(f"meshes/{mesh_name}/{mesh_name}_volumes.png")
    


if __name__=="__main__":
    import sys
    mesh_name = sys.argv[1]
    generate_html(mesh_name)
    plot_ecs_gap_size(mesh_name)
    compute_volumes(mesh_name)

