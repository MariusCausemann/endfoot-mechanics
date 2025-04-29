from itertools import chain

setups = [
    "artnew_cardiac-closed",
    "artnew_cardiac-open",
    #"artnew_cardiac-woAQP4",
    #"artnew_osmotic-pressure",
    #"artnew_osmotic-pressure-aqp4KO",
    #"artnew_cardiac-closed-stiff-PVSx2",
    #"artnew_cardiac-closed-stiff-PVSx4",
    #"artnew_cardiac-closed-stiff-PVSx6",
    #"artnew_cardiac-closed-stiff-PVSx8",
    ]

comparisons = [
    #("artnew", "cardiac-closed", "WT", "artnew", "cardiac-woAQP4", "AQP4-KO"),
    ("artnew", "cardiac-closed", "maximal+resistance","artnew", "cardiac-open", "minimal+resistance"),
    #("artnew", "cardiac-closed", "PVS+stiffness+x1",
    # "artnew", "cardiac-closed-stiff-PVSx2", "PVS+stiffness+x2",
    # "artnew", "cardiac-closed-stiff-PVSx4", "PVS+stiffness+x4",
    # "artnew", "cardiac-closed-stiff-PVSx6", "PVS+stiffness+x6",
    # "artnew", "cardiac-closed-stiff-PVSx8", "PVS+stiffness+x8",),
    #("artnew", "osmotic-pressure", "WT", "artnew", "osmotic-pressure-aqp4KO", "AQP4-KO"),
    #("artnew", "cardiac-closed", "standard", "artnewref", "cardiac-closed", "refined")
    ]

cpus = {"artnew":64, "artnewref":192}

k3d_plots = ["pore_pressure","displacement","vm","k3d_main_arrows"]
k3d_static = ["interface", "boundaries"]

rule all:
    input:
        expand("results/{sim_name}/{sim_name}.xdmf", sim_name=setups),
        expand("results/{sim_name}/plots/{sim_name}_mean_pvs_velocity.png", sim_name=setups),
        expand("results/{sim_name}/{k3d_plot}/{k3d_plot}.html",
               sim_name=setups, k3d_plot=k3d_plots + k3d_static),
        expand("results/{sim_name}/{k3d_plot}_grid.png",
               sim_name=setups, k3d_plot=k3d_plots),
        expand("results/{sim_name}/{sl}", sim_name=setups,
         sl=["flow_slices_1.0", "pressure_slices_1.5", "div_slices_0.5"]),
        expand("results/{sim_name}/animations/{k3d_plot}.{formats}",
               sim_name=setups, k3d_plot=k3d_plots, formats=["gif","apng", "webp", "mp4"]),
        [f"results/comparisons/{'_'.join(chain(*zip(sn[::3],sn[1::3])))}/{'_'.join(sn[2::3])}.png"
        for sn in comparisons], 

rule runSim:
    input:
        config = "config_files/{sim_name}.yml",
        mesh_config = "config_files/{mesh_name}.yml",
        meshfile = "meshes/{mesh_name}/{mesh_name}.xdmf",
        centerline = "meshes/{mesh_name}/centerline.yml",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.hdf",
        outfile="results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    resources:
        ntasks=lambda wildcards: cpus[wildcards.mesh_name],
        time="01:00:00",
        nodes=1,
        mem_mb=int(1),
    threads: lambda wildcards: cpus[wildcards.mesh_name],
    conda:  "envs/fenics_env.yml"
    shell:
        """    
        mpirun -n {resources.ntasks} \
        python3 runEndfeetSimFlux.py \
        -c {wildcards.sim_name} \
        -m  {wildcards.mesh_name}
        """


rule runAnalysisFenics:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.hdf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_fenics.yml",
    conda:  "envs/fenics_env.yml"
    resources:
        ntasks=lambda wildcards: int(cpus[wildcards.mesh_name]/2),
        threads=lambda wildcards: int(cpus[wildcards.mesh_name]/2),
    shell:
        """
        mpirun -n {resources.ntasks} \
        python3 endfeet_buffering_analysis.py {wildcards.sim_name} {wildcards.mesh_name}
        """ 

rule runFlowMetrics:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.hdf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_flow_metrics.yml",
    conda:  "envs/fenics_env.yml"
    resources:
        ntasks=16,
        threads=16,
        time= "02:00:00",
    shell:
        """
        mpirun -n {resources.ntasks} \
        python3 flow_metrics.py {wildcards.sim_name} {wildcards.mesh_name}
        """ 

rule runDispMetrics:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.hdf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_disp_stress_metrics.yml",
    conda:  "envs/fenics_env.yml"
    resources:
        ntasks=16,
        time= "02:00:00",
    shell:
        """
        mpirun -n {resources.ntasks} \
        python3 disp_stress_metrics.py {wildcards.sim_name} {wildcards.mesh_name}
        """ 

rule runAnalysisPyvista:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.hdf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_pyvista.yml",
    conda:  "envs/fenics_env.yml"
    shell:
        """
        python3 evaluate_pv.py {wildcards.sim_name} {wildcards.mesh_name}
        """ 

rule runPeakDistribution:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_peak_distr.yml",
    conda:  "envs/fenics_env.yml"
    shell:
        """
        python3 evaluate_peak_distributions.py {wildcards.sim_name} {wildcards.mesh_name}
        """ 

rule runVisualisation:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        "results/{mesh_name}_{sim_name}/subdomains_sorted.xdmf",
        "results/{mesh_name}_{sim_name}/animations_endfeet3D/endfeet_buffering.gif",
    conda:  "envs/fenics_env.yml"
    shell:
        """
        singularity exec {sing_image} \
        python3 endfeet_buffering_visualisation.py {wildcards.sim_name} {wildcards.mesh_name} && \
        cd results/{wildcards.mesh_name}_{wildcards.sim_name}/animations_endfeet3D &&
        convert -delay 30 *.png endfeet_buffering.gif
        """

rule runVisualisationHTML:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.html",
    conda:  "envs/fenics_env.yml"
    shell:
        """
        python3 endfeet_buffering_visualisation_html.py {wildcards.sim_name} {wildcards.mesh_name}
        """

rule runVisualisationK3D:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        "results/{mesh_name}_{sim_name}/{plot_name}/{plot_name}.html",
    conda:  "envs/k3d_env.yml"
    resources:
        time= "02:00:00",
        partition="hgx2q,dgx2q,milanq",
    shell:
        """
        python3 plotting/endfeet_buffering_visualisation_k3d.py {wildcards.sim_name} {wildcards.mesh_name} {wildcards.plot_name}
        """

rule runVisualisationK3DGrid:
    input:
        "results/{mesh_name}_{sim_name}/{plot_name}/{plot_name}.html",
    output:
        "results/{mesh_name}_{sim_name}/{plot_name}_grid.png",
    conda:  "envs/k3d_env.yml"
    resources:
        time= "00:03:00"
    shell:
        """
        python3 plotting/compose_k3d.py {wildcards.sim_name} {wildcards.mesh_name} {wildcards.plot_name} &&
        python3 plotting/compose_k3d.py {wildcards.sim_name} {wildcards.mesh_name} {wildcards.plot_name} white
        """

rule runVisualisationK3DMovie:
    input:
        "results/{mesh_name}_{sim_name}/{plot_name}/{plot_name}.html",
    output:
        "results/{mesh_name}_{sim_name}/animations/{plot_name}.{format}",
    conda:  "envs/k3d_env.yml"
    wildcard_constraints:
        format="apng|webp|gif"
    params:
        file = "results/{mesh_name}_{sim_name}/animations/{plot_name}"
    resources:
        time= "00:03:00"
    shell:
        """
        python3 -c "from plotting.headless import write_animation
img_dir = 'results/{wildcards.mesh_name}_{wildcards.sim_name}/{wildcards.plot_name}'
write_animation(img_dir, '{output[0]}', 10)"
        """

rule runVisualisationK3DMP4:
    input:
        "results/{mesh_name}_{sim_name}/animations/{plot_name}.apng",
    output:
        "results/{mesh_name}_{sim_name}/animations/{plot_name}.mp4",
    conda:  "envs/k3d_env.yml"
    resources:
        time= "00:03:00"
    shell:
        "ffmpeg -y -i {input[0]} {output[0]}"


rule runMeshDeformation:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_ts.yml",
    conda:  "envs/fenics_env.yml"
    resources:
        time= "06:00:00",
        ntasks=16,
    shell:
        """
        python3 evaluate_mesh_deformation.py {wildcards.sim_name} {wildcards.mesh_name}
        """

rule generatePlots:
    input:
        #"results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_fenics.yml",
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_results_pyvista.yml",
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_peak_distr.yml",
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_ts.yml",
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_flow_metrics.yml",
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}_disp_stress_metrics.yml",
    output:
        "results/{mesh_name}_{sim_name}/plots/{mesh_name}_{sim_name}_mean_pvs_velocity.png",
    conda:  "envs/fenics_env.yml"
    resources:
        time="00:05:00"
    shell:
        """
        mkdir -p results/{wildcards.mesh_name}_{wildcards.sim_name}/plots &&
        ls plotting/*/*.py|xargs -n 4 -P 4 -I % python3 % {wildcards.sim_name} {wildcards.mesh_name}
        """

group2 = lambda ls: list(zip(ls[::2], ls[1::2]))

def scenarios2cmd(scenarios, scenario_names):
    sc = group2(scenarios.split("_"))
    sn = scenario_names.split("_")
    cmd = " ".join(f"{mn} {sn} {n}" for (sn,mn), n in zip(sc, sn))
    return cmd

rule compareScenarios:
    input:
        lambda wildcards: [f"results/{mn}_{sn}/{mn}_{sn}_flow_metrics.yml" for mn, sn in group2(wildcards.scenarios.split("_"))],
        lambda wildcards: [f"results/{mn}_{sn}/{mn}_{sn}_disp_stress_metrics.yml" for mn, sn in group2(wildcards.scenarios.split("_"))],
        lambda wildcards: [f"results/{mn}_{sn}/{mn}_{sn}_ts.yml" for mn, sn in group2(wildcards.scenarios.split("_"))],
        lambda wildcards: [f"results/{mn}_{sn}/{mn}_{sn}_results_pyvista.yml" for mn, sn in group2(wildcards.scenarios.split("_"))]

    output:
        "results/comparisons/{scenarios}/{scenario_names}.png",
    conda:  "envs/fenics_env.yml"
    params:
        cmd=lambda wildcards: scenarios2cmd(wildcards.scenarios, wildcards.scenario_names)
    shell:
        """
        python3 plotting/compare_scenarios.py \
        {params.cmd}
        """


def get_input_files(wildcards):
    return expand("results/{s}/{s}_flow_metrics.yml", s=wildcards.scenarios1.split("*"))


rule compareMultipleScenarios:
    input:
        get_input_files
    output:
        plot="results/comparisons/{scenarios1}+{scenarios2}/{name1}_{name2}.png",
         #"results/comparisons/{mesh_name1}_{sim_name1}_{mesh_name2}_{sim_name2}/{name1}_{name2}_peak.png"
    conda:  "envs/fenics_env.yml"
    params:
        scenarios1 = lambda wildcards: wildcards.scenarios1.split("*"),
        scenarios2 = lambda wildcards: wildcards.scenarios2.split("*")
    shell:
        """
        python3 plotting/compare_multiple_scenarios.py \
        -s1 {params.scenarios1} \
        -s2 {params.scenarios2} \
        -n1 {wildcards.name1} -n2 {wildcards.name2} \
        -f {output.plot}
        """

rule plotPressureSlices:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        directory("results/{mesh_name}_{sim_name}/pressure_slices_{cmax}/")
    conda:  "envs/fenics_env.yml"
    shell:
        "python3 plotting/pressure_warps.py {wildcards.sim_name} {wildcards.mesh_name} {wildcards.cmax}"


rule plotDivSlices:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        directory("results/{mesh_name}_{sim_name}/div_slices_{cmax}/")
    conda:  "envs/fenics_env.yml"
    shell:
        "python3 plotting/divd_warps.py {wildcards.sim_name} {wildcards.mesh_name} {wildcards.cmax}"


rule plotFlowSlices:
    input:
        "results/{mesh_name}_{sim_name}/{mesh_name}_{sim_name}.xdmf",
    output:
        directory("results/{mesh_name}_{sim_name}/flow_slices_{cmax}/")
    conda:  "envs/fenics_env.yml"
    shell:
        "python3 plotting/flow_slices.py {wildcards.sim_name} {wildcards.mesh_name} {wildcards.cmax}"











    
