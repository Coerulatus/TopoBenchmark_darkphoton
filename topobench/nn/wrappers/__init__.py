"""This module implements the wrappers for the neural networks."""

from topobench.nn.wrappers.base import AbstractWrapper
from topobench.nn.wrappers.cell import (
    CANWrapper,
    CCCNWrapper,
    CCXNWrapper,
    CWNWrapper,
)
from topobench.nn.wrappers.combinatorial import TuneWrapper
from topobench.nn.wrappers.graph import (
    GNNWrapper,
    GraphMLPWrapper,
    MLPWrapper,
)
from topobench.nn.wrappers.hypergraph import HypergraphWrapper
from topobench.nn.wrappers.simplicial import (
    SANWrapper,
    SCCNNWrapper,
    SCCNWrapper,
    SCNWrapper,
)

# ... import other readout classes here
# For example:
# from topobenchmark.nn.wrappers.other_wrapper_1 import OtherWrapper1
# from topobenchmark.nn.wrappers.other_wrapper_2 import OtherWrapper2


# Export all wrappers
__all__ = [
    "AbstractWrapper",
    "CANWrapper",
    "CCCNWrapper",
    "CCXNWrapper",
    "CWNWrapper",
    "GNNWrapper",
    "GraphMLPWrapper",
    "HypergraphWrapper",
    "MLPWrapper",
    "SANWrapper",
    "SCCNNWrapper",
    "SCCNWrapper",
    "SCNWrapper",
    "TuneWrapper",
    # "OtherWrapper1",
    # "OtherWrapper2",
    # ... add other readout classes here
]
