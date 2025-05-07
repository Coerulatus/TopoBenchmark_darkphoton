"""Run experiments on the dark_photons dataset."""

import subprocess

# Configurations
project_name = "test"
experiments = [
    "mlp",
    "gat",
    "detector_layer_no_maxdiff",
    "detector_layer",
    "same_particle_no_maxdiff",
    "same_particle",
]  # ["gat", "detector_layer_no_maxdiff"]#, "detector_layer"]#, "same_particle"]  # "mlp", "gat",
checkpoints = [
    "/home/marco/Documents/phd/TopoBenchmark_darkphoton/logs/train/multiruns/mlp/0/checkpoints/epoch_098.ckpt",
    "/home/marco/Documents/phd/TopoBenchmark_darkphoton/logs/train/multiruns/gat/0/checkpoints/epoch_099.ckpt",
    "/home/marco/Documents/phd/TopoBenchmark_darkphoton/logs/train/multiruns/detector_layer_no_maxdiff/0/checkpoints/epoch_092.ckpt",
    "/home/marco/Documents/phd/TopoBenchmark_darkphoton/logs/train/multiruns/detector_layer/0/checkpoints/epoch_092.ckpt",
    "/home/marco/Documents/phd/TopoBenchmark_darkphoton/logs/train/multiruns/same_particle_no_maxdiff/0/checkpoints/epoch_099.ckpt",
    "/home/marco/Documents/phd/TopoBenchmark_darkphoton/logs/train/multiruns/same_particle/0/checkpoints/epoch_095.ckpt",
]
# Run the commands in Python
for experiment, checkpoint in zip(experiments, checkpoints, strict=False):
    command = [
        "python",
        "-m",
        "topobench.logits",
        f"experiment=dark_photons/{experiment}",
        f"logger.wandb.project={project_name}",
        "dataset.split_params.data_seed=0",
        f"ckpt_path={checkpoint}",
        f"+output_path=./logs/logits/{experiment}",
    ]

    print(f"Running: {' '.join(command)}")
    subprocess.run(command)
