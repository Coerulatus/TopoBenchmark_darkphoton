"""Abstract class for lifting pointclouds to graphs."""

from topobench.transforms.liftings.liftings import PointCloudLifting


class PointCloud2GraphLifting(PointCloudLifting):
    r"""Abstract class for lifting pointclouds to graphs.

    Parameters
    ----------
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = "pointcloud2graph"
