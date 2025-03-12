"""Run experiments on the dark_photons dataset."""

import subprocess

# Configurations
project_name = "dark_photons"
experiments = ["mlp", "gat", "detector_layer_hypergraph"]

data_seeds = [4, 3, 2, 1, 0]
# Run the commands in Python
for experiment in experiments:
    for data_seed in data_seeds:
        command = [
            "python",
            "-m",
            "topobenchmark",
            f"experiment=dark_photons/{experiment}",
            f"logger.wandb.project={project_name}",
            f"dataset.split_params.data_seed={data_seed}",
        ]

        print(f"Running: {' '.join(command)}")
        subprocess.run(command)
