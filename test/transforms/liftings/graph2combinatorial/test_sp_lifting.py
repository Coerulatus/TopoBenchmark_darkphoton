import torch
import pytest
import networkx as nx
from modules.transforms.liftings.graph2combinatorial.sp_lifting import \
    DirectedFlagComplex, SPLifting

class TestDirectedFlagComplex():

    @pytest.fixture
    def digraph(self):
        """Creates a simple directed graph for testing."""
        digraph = nx.DiGraph()
        edges = [(i, i + offset) for i in range(100) for offset in (1, 2)]
        digraph.add_edges_from(edges)
        return digraph

    @pytest.fixture
    def dfc(self, digraph):
        """Initializes a DirectedFlagComplex instance for testing."""
        return DirectedFlagComplex(digraph=digraph, complex_dim=2,
                                   flagser_num_threads=4)

    def test_initialization(self, dfc, digraph):
        """Test the initialization of the DirectedFlagComplex."""
        assert dfc.digraph == digraph
        assert dfc.complex_dim == 2

    def test_d_i_batched(self, dfc):
        """Test the _d_i_batched method, that computes the i-th boundary
        operator."""
        simplices = torch.tensor([[0, 1, 2], [1, 2, 3]], device=dfc.device)
        result = dfc._d_i_batched(1, simplices)
        expected = torch.tensor([[0, 2], [1, 3]], device=dfc.device)
        assert torch.equal(result, expected)

    def test_gen_q_faces_batched(self, dfc):
        """Test the _gen_q_faces_batched method which generates the q-faces"""
        simplices = torch.tensor([[0, 1, 2], [1, 2, 3]], device=dfc.device)
        result = dfc._gen_q_faces_batched(simplices, 2)
        expected = torch.tensor([[[0, 1], [0, 2], [1, 2]],
                                 [[1, 2], [1, 3], [2, 3]]],
                                device=dfc.device)
        assert torch.equal(result, expected)

    def test_multiple_contained_chunked(self, dfc):
        """Test the _multiple_contained_chunked method which computes the
        containment of multiple simplices in multiple simplices."""
        sigmas = torch.tensor([[0, 1], [1, 2]], device= dfc.device)
        taus = torch.tensor([[0, 1, 2], [1, 2, 3]], device=dfc.device)
        result = dfc._multiple_contained_chunked(sigmas, taus)
        #[0,1] is contained in [0,1,2] and [1,2] is  contained in [0,1,
        # 2]  and [1,2,3]
        expected_indices = torch.tensor([[0, 1, 1],
                       [0, 0, 1]], dtype=torch.long)
        expected = torch.sparse_coo_tensor(expected_indices,
                                           torch.ones(3, dtype=torch.bool),
                                           size=(2, 2), device="cpu")
        assert torch.equal(result._indices(), expected._indices())
        assert torch.equal(result._values(), expected._values())

    def test_alpha_q_contained_sparse(self, dfc):
        """Test the _alpha_q_contained_sparse method"""
        sigmas = torch.tensor([[0, 1, 2], [1, 2, 3]], device=dfc.device)
        taus = torch.tensor([[0, 1, 2], [1, 2, 3]], device=dfc.device)
        result = dfc._alpha_q_contained_sparse(sigmas, taus, 1)
        expected_indices = torch.tensor([[0, 0, 1, 1],
                       [0, 1, 0, 1]], dtype=torch.long)
        expected = torch.sparse_coo_tensor(expected_indices,  torch.ones(4,
                                            dtype=torch.bool),  size=(2, 2),
                                           device="cpu")

        assert torch.equal(result._indices(), expected._indices())
        assert torch.equal(result._values(), expected._values())

    def test_qij_adj(self, dfc):
        """Test the qij_adj method."""
        result = dfc.qij_adj(dfc.complex[2], dfc.complex[2], q=1, i=0, j=2)
        expected_indices = torch.tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,
                                           4,  4,  5,  5,  6,  6,  7,  7,
                                           8,  8,
          9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
         18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26,
         27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35,
         36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44,
         45, 45, 46, 46, 47, 47, 48, 48, 49, 49, 50, 50, 51, 51, 52, 52, 53, 53,
         54, 54, 55, 55, 56, 56, 57, 57, 58, 58, 59, 59, 60, 60, 61, 61, 62, 62,
         63, 63, 64, 64, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71,
         72, 72, 73, 73, 74, 75, 75, 76, 76, 77, 77, 78, 78, 79, 79, 80, 80, 81,
         81, 82, 82, 83, 83, 84, 84, 85, 85, 86, 86, 87, 87, 88, 88, 89, 89, 90,
         90, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96, 96, 97, 97, 98, 98],
        [ 0, 25,  1, 26,  2, 27,  3, 28,  4, 29,  5, 30,  6, 31,  7, 32,  8, 33,
          9, 34, 10, 35, 11, 36, 12, 37, 13, 38, 14, 39, 15, 40, 16, 41, 17, 42,
         18, 43, 19, 44, 20, 45, 21, 46, 22, 47, 23, 48, 24, 49, 25, 50, 26, 51,
         27, 52, 28, 53, 29, 54, 30, 55, 31, 56, 32, 57, 33, 58, 34, 59, 35, 60,
         36, 61, 37, 62, 38, 63, 39, 64, 40, 65, 41, 66, 42, 67, 43, 68, 44, 69,
         45, 70, 46, 71, 47, 72, 48, 73, 49, 74, 50, 75, 51, 76, 52, 77, 53, 78,
         54, 79, 55, 80, 56, 81, 57, 82, 58, 83, 59, 84, 60, 85, 61, 86, 62, 87,
         63, 88, 64, 89, 65, 90, 66, 91, 67, 92, 68, 93, 69, 94, 70, 95, 71, 96,
         72, 97, 73, 98, 74,  1, 75,  2, 76,  3, 77,  4, 78,  5, 79,  6, 80,  7,
         81,  8, 82,  9, 83, 10, 84, 11, 85, 12, 86, 13, 87, 14, 88, 15, 89, 16,
         90, 17, 91, 18, 92, 19, 93, 20, 94, 21, 95, 22, 96, 23, 97, 24,
          98]],  dtype=torch.long)
        assert torch.equal(result, expected_indices)



class TestSPLifting():
    def test_sp_lifting(self):
        pass