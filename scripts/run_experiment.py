"""Run experiments on the dark_photons dataset."""

import subprocess

# Configurations
project_name = "dark_photons_roc"
experiments = [
    "same_particle_no_maxdiff"
]  # ["gat", "detector_layer_no_maxdiff"]#, "detector_layer"]#, "same_particle"]  # "mlp", "gat",

data_seeds = [0]
# Run the commands in Python
for experiment in experiments:
    for data_seed in data_seeds:
        command = [
            "python",
            "-m",
            "topobench",
            f"experiment=dark_photons/{experiment}",
            f"logger.wandb.project={project_name}",
            f"dataset.split_params.data_seed={data_seed}",
        ]

        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
