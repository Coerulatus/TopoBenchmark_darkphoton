"""Loaders for Dark Photon datasets."""

from omegaconf import DictConfig
from torch_geometric.data import Dataset

from topobenchmark.data.loaders.base import AbstractLoader
from topobenchmark.data.datasets import DarkPhotonDataset


class DarkPhotonDatasetLoader(AbstractLoader):
    """Load Dark Photon datasets.

    Parameters
    ----------
    parameters : DictConfig
        Configuration parameters containing:
            - data_dir: Root directory for data.
            - url: URL to download the dataset.
            - subset: Percentage of the dataset to load.
            - verbose: Verbosity level.
    """

    def __init__(self, parameters: DictConfig) -> None:
        super().__init__(parameters)

    def load_dataset(self) -> Dataset:
        """Load Dark Photon dataset.

        Returns
        -------
        Dataset
            The loaded Dark Photon dataset.

        Raises
        ------
        RuntimeError
            If dataset loading fails.
        """

        dataset = DarkPhotonDataset(
            root=str(self.root_data_dir),
            **self.parameters,
        )
        return dataset
