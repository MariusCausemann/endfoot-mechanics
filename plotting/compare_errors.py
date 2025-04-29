import sys, os
import numpy as np
import matplotlib.pyplot as plt
import yaml

if __name__ == '__main__':
    sim_name1 = sys.argv[1]
    mesh_name1 = sys.argv[2]
    sim_name2 = sys.argv[3]
    mesh_name2 = sys.argv[4]

    variables = ["d", "pP_i","pP_e","phi_i", "phi_e"]
    name1 = f"{mesh_name1}_{sim_name1}"
    name2 = f"{mesh_name2}_{sim_name2}"

    fdir = f"results/comparisons/{name1}_{name2}"
    with open(f"{fdir}/errornorms.yml", 'r') as file:
        errors = yaml.load(file, Loader=yaml.FullLoader)

    times = errors.pop("times")

    for norm, err in errors.items():
        for var, ts in err.items():
            plt.figure()
            for model, res in ts.items():
                plt.plot(times, res, label=f"{var} {model}")
            plt.title(f"{norm} {var}")
            plt.legend()
            plt.ylabel(f"{norm} norm")
            plt.xlabel("t [s]")
            plt.savefig(f"{fdir}/{norm}_{var}.png")

    relative_errors = {}
    for norm, err in errors.items():
        relative_errors[norm] = {}
        for var, data in err.items():
            rel_err = np.array(data["err"]) / np.array(data[name2])
            rel_err = np.nan_to_num(rel_err)
            relative_errors[norm][var] = rel_err
    
    for norm, err in relative_errors.items():
        plt.figure()
        for var, data in err.items():
            plt.plot(times, data*100, ".-", label=var)
        plt.legend()
        plt.ylabel(f"relative {norm} error [%]")
        plt.xlabel("t [s]")
        plt.savefig(f"{fdir}/relative_{norm}.png")







