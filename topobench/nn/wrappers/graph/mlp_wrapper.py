"""Wrapper for the MLP models."""

from topobench.nn.wrappers.base import AbstractWrapper


class MLPWrapper(AbstractWrapper):
    r"""Wrapper for the MLP models.

    This wrapper defines the forward pass of the model. The MLP models return
    the embeddings of the cells of rank 0.
    """

    def forward(self, batch):
        r"""Forward pass for the MLP wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing the updated model output.
        """

        x_0 = self.backbone(
            batch.x_0,
        )

        model_out = {"labels": batch.y, "batch_0": batch.batch_0}
        model_out["x_0"] = x_0

        return model_out
