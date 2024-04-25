import torch
import torch_geometric

from modules.transforms.liftings.graph2hypergraph.base import Graph2HypergraphLifting


class HypergraphKHopLifting(Graph2HypergraphLifting):
    r"""Lifts graphs to hypergraph domain by considering k-hop neighborhoods.

    Parameters
    ----------
    k_value : int, optional
        The number of hops to consider. Default is 1.
    **kwargs : optional
        Additional arguments for the class.
    """

    def __init__(self, k_value=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k_value

    def lift_topology(self, data: torch_geometric.data.Data) -> dict:
        r"""Lifts the topology of a graph to hypergraph domain by considering k-hop neighborhoods.

        Parameters
        ----------
        data : torch_geometric.data.Data
            The input data to be lifted.

        Returns
        -------
        dict
            The lifted topology.
        """
        # Check if data has instance x:
        if hasattr(data, "x") and data.x is not None:
            num_nodes = data.x.shape[0]
        else:
            num_nodes = data.num_nodes

        incidence_1 = torch.zeros(num_nodes, num_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)

        # Detect isolated nodes
        isolated_nodes = [i for i in range(num_nodes) if i not in edge_index[0]]
        if len(isolated_nodes) > 0:
            # Add completely isolated nodes to the edge_index
            edge_index = torch.cat(
                [
                    edge_index,
                    torch.tensor([isolated_nodes, isolated_nodes], dtype=torch.long),
                ],
                dim=1,
            )

        for n in range(num_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1

        num_hyperedges = incidence_1.shape[1]
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        return {
            "incidence_hyperedges": incidence_1,
            "num_hyperedges": num_hyperedges,
            "x_0": data.x,
        }
