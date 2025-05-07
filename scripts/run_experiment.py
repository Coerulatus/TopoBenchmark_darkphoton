"""Run experiments on the dark_photons dataset."""

import subprocess

# Configurations
project_name = "dark_photons_v2"
experiments = [
    "mlp"
]  # ["gat", "detector_layer_no_maxdiff"]#, "detector_layer"]#, "same_particle"]  # "mlp", "gat",

# Run the commands in Python
for experiment in experiments:
    command = [
        "python",
        "-m",
        "topobench",
        f"experiment=dark_photons/{experiment}",
        f"logger.wandb.project={project_name}",
        "dataset.split_params.data_seed=3,4",
        "--multirun",
    ]

    print(f"Running: {' '.join(command)}")
    subprocess.run(command)
