"""Fully connected graph lifting."""

import torch

from topobenchmark.transforms.liftings.pointcloud2graph import (
    PointCloud2GraphLifting,
)


class FullyConnectedLifting(PointCloud2GraphLifting):
    """Fully connected graph lifting.

    Parameters
    ----------
    **kwargs : dict
        Additional key-word arguments passed to the super class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data):
        """Forward pass.

        Parameters
        ----------
        data : torch_geometric.data.Data
            Data object.

        Returns
        -------
        torch_geometric.data.Data
            Data object with the fully connected graph.
        """
        edge_index = (
            torch.combinations(
                torch.arange(data.x.size(0)), with_replacement=True
            )
            .t()
            .contiguous()
        )
        edge_attr = None
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data
