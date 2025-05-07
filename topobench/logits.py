"""Main entry point for training and testing models."""

import os
import random
from typing import Any

import hydra
import lightning as L
import numpy as np
import rootutils
import torch
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import TBDataloader
from topobench.utils import (
    RankedLogger,
    extras,
)
from topobench.utils.config_resolvers import (
    get_default_metrics,
    get_default_transform,
    get_monitor_metric,
    get_monitor_mode,
    get_required_lifting,
    infer_in_channels,
    infer_num_cell_dimensions,
)

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #


OmegaConf.register_new_resolver(
    "get_default_metrics", get_default_metrics, replace=True
)
OmegaConf.register_new_resolver(
    "get_default_transform", get_default_transform, replace=True
)
OmegaConf.register_new_resolver(
    "get_required_lifting", get_required_lifting, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_metric", get_monitor_metric, replace=True
)
OmegaConf.register_new_resolver(
    "get_monitor_mode", get_monitor_mode, replace=True
)
OmegaConf.register_new_resolver(
    "infer_in_channels", infer_in_channels, replace=True
)
OmegaConf.register_new_resolver(
    "infer_num_cell_dimensions", infer_num_cell_dimensions, replace=True
)
OmegaConf.register_new_resolver(
    "parameter_multiplication", lambda x, y: int(int(x) * int(y)), replace=True
)


def initialize_hydra() -> DictConfig:
    """Initialize Hydra when main is not an option (e.g. tests).

    Returns
    -------
    DictConfig
        A DictConfig object containing the config tree.
    """
    hydra.initialize(
        version_base="1.3", config_path="../configs", job_name="run"
    )
    cfg = hydra.compose(config_name="run.yaml")
    return cfg


torch.set_num_threads(1)
log = RankedLogger(__name__, rank_zero_only=True)


def save_model_outputs(model, dataloader, output_folder):
    """Run the model on the given dataloader and saves logits and true labels as a tensor.

    Parameters
    ----------
    model : LightningModule
        The model to be evaluated.
    dataloader : DataLoader
        The dataloader containing the test dataset.
    output_folder : str
        The folder where the output tensor will be saved.

    Returns
    -------
    str
        The path to the saved output tensor.
    """

    model.eval()  # Set model to evaluation mode
    device = next(model.parameters()).device  # Get model device
    all_logits, all_labels = [], []

    with torch.no_grad():  # Disable gradient computation for inference
        for batch in dataloader:
            labels = batch.y  # Assuming batch contains (inputs, labels)
            batch = batch.to(device)
            model_input = model.feature_encoder(batch)
            backbone_out = model.backbone(model_input)
            model_out = model.readout(backbone_out, batch)
            logits = model_out["logits"]
            all_logits.append(logits)
            all_labels.append(labels.cpu())

    # Stack results
    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0).unsqueeze(1)

    # Create output tensor of shape (2, n)
    output_tensor = torch.cat([logits_tensor, labels_tensor], dim=1)

    # Ensure unique filename
    os.makedirs(output_folder, exist_ok=True)
    index = 0
    while os.path.exists(
        os.path.join(output_folder, f"model_output_{index}.pt")
    ):
        index += 1
    output_path = os.path.join(output_folder, f"model_output_{index}.pt")

    # Save tensor
    torch.save(output_tensor, output_path)
    print(f"Saved output tensor to {output_path}")

    return output_path


def run(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Can additionally evaluate on a testset, using best weights obtained during training.

    This method is wrapped in optional @task_wrapper decorator, that controls
    the behavior during failure. Useful for multiruns, saving info about the
    crash, etc.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        A tuple with metrics and dict with all instantiated objects.
    """
    # Set seed for random number generators in pytorch, numpy and python.random
    # if cfg.get("seed"):
    L.seed_everything(cfg.seed, workers=True)
    # Seed for torch
    torch.manual_seed(cfg.seed)
    # Seed for numpy
    np.random.seed(cfg.seed)
    # Seed for python random
    random.seed(cfg.seed)

    # Instantiate and load dataset
    log.info(f"Instantiating loader <{cfg.dataset.loader._target_}>")
    dataset_loader = hydra.utils.instantiate(cfg.dataset.loader)
    dataset, dataset_dir = dataset_loader.load()
    # Preprocess dataset and load the splits
    log.info("Instantiating preprocessor...")
    transform_config = cfg.get("transforms", None)
    preprocessor = PreProcessor(dataset, dataset_dir, transform_config)
    dataset_train, dataset_val, dataset_test = (
        preprocessor.load_dataset_splits(cfg.dataset.split_params)
    )
    # Prepare datamodule
    log.info("Instantiating datamodule...")
    if cfg.dataset.parameters.task_level in ["node", "graph"]:
        datamodule = TBDataloader(
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
            **cfg.dataset.get("dataloader_params", {}),
        )
    else:
        raise ValueError("Invalid task_level")

    # Model for us is Network + logic: inputs backbone, readout, losses
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(
        cfg.model,
        evaluator=cfg.evaluator,
        optimizer=cfg.optimizer,
        loss=cfg.loss,
    )

    if not cfg.get("ckpt_path"):
        raise ValueError("No ckpt_path provided!")
    ckpt_path = cfg.ckpt_path
    log.info(
        f"Attempting to load weights from the provided ckpt_path: {ckpt_path}"
    )
    try:
        # trainer.test(
        #     model=model, datamodule=datamodule, ckpt_path=ckpt_path
        # )
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["state_dict"])
        test_dataloader = datamodule.test_dataloader()
        ckpt_folder = os.path.dirname(ckpt_path)
        output_path = cfg.get("output_path", ckpt_folder)
        save_model_outputs(model, test_dataloader, output_path)

    except FileNotFoundError:
        log.warning(
            f"No checkpoint file found at the provided ckpt_path: {ckpt_path}."
        )
        log.info("Trying with best model instead...")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="run.yaml"
)
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    Parameters
    ----------
    cfg : DictConfig
        Configuration composed by Hydra.

    Returns
    -------
    float | None
        Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    run(cfg)


if __name__ == "__main__":
    main()
