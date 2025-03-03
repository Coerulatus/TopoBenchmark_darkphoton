"""Run experiments on the dark_photons dataset."""

import subprocess

# Configurations
project_name = "dark_photons_test"
experiment = "mlp"

# Run the commands in Python
command = [
    "python",
    "-m",
    "topobenchmark",
    f"experiment=dark_photons/{experiment}",
    f"logger.wandb.project={project_name}",
]

print(f"Running: {' '.join(command)}")
subprocess.run(command)
