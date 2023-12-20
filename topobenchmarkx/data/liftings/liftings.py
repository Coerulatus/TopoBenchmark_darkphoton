import copy
from itertools import combinations

import numpy as np
import torch
import torch_geometric
from scipy.optimize import minimize
import networkx as nx

from topobenchmarkx.data.liftings.lifting import AbstractLifting


class HypergraphKHopLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.added_fields = ["hyperedges"]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(
                n, self.k, edge_index
            )
            incidence_1[n, neighbors] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        data.__setitem__(self.added_fields[0], incidence_1)
        return data


class HypergraphKNearestNeighborsLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, k=1):
        super().__init__()
        self.transform = torch_geometric.transforms.KNNGraph(k)
        self.added_fields = ["hyperedges"]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        data_lifted = copy.copy(data)
        data_lifted.pos = data_lifted.x
        n_nodes = data.x.shape[0]
        incidence_1 = torch.zeros(n_nodes, n_nodes)
        data_lifted = self.transform(data_lifted)
        incidence_1[data_lifted.edge_index[0], data_lifted.edge_index[1]] = 1
        incidence_1 = torch.Tensor(incidence_1).to_sparse_coo()
        data.__setitem__(self.added_fields[0], incidence_1)
        return data


class SimplicialNeighborhoodLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, complex_dim=2):
        super().__init__()
        self.complex_dim = complex_dim
        self.added_fields = []
        for i in range(1, complex_dim + 1):
            self.added_fields += [
                f"incidence_{i}",
                f"laplacian_down_{i}",
                f"laplacian_up_{i}",
            ]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        simplices = [set() for _ in range(self.complex_dim + 1)]
        for n in range(n_nodes):
            neighbors, _, _, _ = torch_geometric.utils.k_hop_subgraph(n, 1, edge_index)
            if n not in neighbors:
                neighbors.append(n)
            neighbors = neighbors.numpy()
            neighbors = set(neighbors)
            for i in range(self.complex_dim + 1):
                for c in combinations(neighbors, i + 1):
                    simplices[i].add(tuple(c))

        for i in range(self.complex_dim + 1):
            simplices[i] = list(simplices[i])
        incidences = [torch.zeros(len(simplices[i]), len(simplices[i+1])) for i in range(self.complex_dim)]
        laplacians_up = [torch.zeros(len(simplices[i]), len(simplices[i])) for i in range(self.complex_dim)]
        laplacians_down = [torch.zeros(len(simplices[i+1]), len(simplices[i+1])) for i in range(self.complex_dim)]
        for i in range(self.complex_dim):
            for idx_i, s_i in enumerate(simplices[i]):
                for idx_i_1, s_i_1 in enumerate(simplices[i + 1]):
                    if all(e in s_i_1 for e in s_i):
                        incidences[i][idx_i][idx_i_1] = 1
            degree = torch.diag(torch.sum(incidences[i], dim=1))
            laplacians_up[i] = 2 * degree - torch.mm(
                incidences[i], torch.transpose(incidences[i], 1, 0)
            )
            degree = torch.diag(torch.sum(incidences[i], dim=0))
            laplacians_down[i] = 2 * degree - torch.mm(
                torch.transpose(incidences[i], 1, 0), incidences[i]
            )

        for i, field in enumerate(self.added_fields):
            if i % 3 == 0:
                data.__setitem__(field, incidences[int(i / 3)].to_sparse_coo())
            if i % 3 == 1:
                data.__setitem__(field, laplacians_up[int(i / 3)].to_sparse_coo())
            if i % 3 == 2:
                data.__setitem__(field, laplacians_down[int(i / 3)].to_sparse_coo())
        return data
    
class CellCyclesLifting(torch_geometric.transforms.BaseTransform):
    def __init__(self, aggregation="sum"):
        super().__init__()
        self.added_fields = []
        if not aggregation in ["sum"]:
            raise NotImplementedError
        self.aggregation = aggregation
        self.added_fields = ["x_1", 
                             "incidence_1", 
                             "laplacian_down_1", 
                             "laplacian_up_1",
                             "incidence_2", 
                             "laplacian_down_2", 
                             "laplacian_up_2"]

    def forward(self, data: torch_geometric.data.Data) -> dict:
        n_nodes = data.x.shape[0]
        # edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        edges = [(i.item(),j.item()) for i, j in zip(data.edge_index[0], data.edge_index[1])]
        G = nx.Graph()
        G.add_edges_from(edges)
        cycles = nx.cycle_basis(G)
        n_edges = len(edges)
        n_cells = len(cycles)
        incidence_1 = torch.zeros([n_nodes, n_edges])
        incidence_2 = torch.zeros([n_edges, n_cells])
        edges = [set(e) for e in edges]
        for i, edge in enumerate(edges):
            incidence_1[list(edge), i] = 1
        for i, cycle in enumerate(cycles):
            for j in range(len(cycle)):
                if j==len(cycle)-1:
                    edge = {cycle[j],cycle[0]}
                else:
                    edge = {cycle[j],cycle[j+1]}
                incidence_2[edges.index(edge), i] = 1
        degree = torch.diag(torch.sum(incidence_1, dim=1))
        laplacian_up_1 = 2 * degree - torch.mm(
            incidence_1, torch.transpose(incidence_1, 1, 0)
        )
        degree = torch.diag(torch.sum(incidence_2, dim=1))
        laplacian_up_2 = 2 * degree - torch.mm(
            incidence_2, torch.transpose(incidence_2, 1, 0)
        )
        degree = torch.diag(torch.sum(incidence_1, dim=0))
        laplacian_down_1 = 2 * degree - torch.mm(
            torch.transpose(incidence_1, 1, 0), incidence_1
        )
        degree = torch.diag(torch.sum(incidence_2, dim=0))
        laplacian_down_2 = 2 * degree - torch.mm(
            torch.transpose(incidence_2, 1, 0), incidence_2
        )
        
        if self.aggregation=="sum":
            x_1 = torch.mm(torch.transpose(incidence_1,1,0),data.x)
                
        data.__setitem__("incidence_1",incidence_1.to_sparse_coo())
        data.__setitem__("incidence_2",incidence_2.to_sparse_coo())
        data.__setitem__("laplacian_up_1",laplacian_up_1.to_sparse_coo())
        data.__setitem__("laplacian_up_2",laplacian_up_2.to_sparse_coo())
        data.__setitem__("laplacian_down_2",laplacian_down_2.to_sparse_coo())
        data.__setitem__("laplacian_down_1",laplacian_down_1.to_sparse_coo())
        data.__setitem__("x_1", x_1)
        
        return data