"""Utilities for logging hyperparameters."""

from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from topobench.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    r"""Control which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    Parameters
    ----------
    object_dict : dict[str, Any]
        A dictionary containing the following objects:
            - `"cfg"`: A DictConfig object containing the main config.
            - `"model"`: The Lightning model.
            - `"trainer"`: The Lightning trainer.
    """
    hparams = {}

    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    for key in cfg:
        hparams[key] = cfg[key]

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)
