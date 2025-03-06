"""Abstract class for lifting graphs to pointcloud complexes."""

from topobenchmark.transforms.liftings.liftings import GraphLifting


class Graph2PointcloudLifting(GraphLifting):
    r"""Abstract class for lifting graphs to pointclouds.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "graph2pointcloud"
