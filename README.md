<h2 align="center">
  <img src="resources/logo.jpg" width="800">
</h2>

<h3 align="center">
    A Comprehensive Benchmark Suite for Topological Deep Learning
</h3>

<p align="center">
Assess how your model compares against state-of-the-art topological neural networks.
</p>

<div align="center">

[![Lint](https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml/badge.svg)](https://github.com/geometric-intelligence/TopoBench/actions/workflows/lint.yml)
[![Test](https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml/badge.svg)](https://github.com/geometric-intelligence/TopoBench/actions/workflows/test.yml)
[![Codecov](https://codecov.io/gh/geometric-intelligence/TopoBench/branch/main/graph/badge.svg)](https://app.codecov.io/gh/geometric-intelligence/TopoBench)
[![Docs](https://img.shields.io/badge/docs-website-brightgreen)](https://geometric-intelligence.github.io/topobench/index.html)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/geometric-intelligence/TopoBench?color=green)](https://github.com/geometric-intelligence/TopoBench/blob/main/LICENSE)
[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/geometric-intelligenceworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)


</div>

<p align="center">
  <a href="#pushpin-overview">Overview</a> •
  <a href="#jigsaw-get-started">Get Started</a> •
  <a href="#anchor-tutorials">Tutorials</a> •
  <a href="#gear-neural-networks">Neural Networks</a> •
  <a href="#rocket-liftings">Liftings</a> •
  <a href="#books-datasets">Datasets</a> •
  <a href="#mag-references">References</a> 
</p>



## :pushpin: Overview

`TopoBench` (TB) is a modular Python library designed to standardize benchmarking and accelerate research in Topological Deep Learning (TDL). In particular, TB allows to train and compare the performances of all sorts of Topological Neural Networks (TNNs) across the different topological domains, where by _topological domain_ we refer to a graph, a simplicial complex, a cellular complex, or a hypergraph. For detailed information, please refer to the [`TopoBench: A Framework for Benchmarking Topological Deep Learning`](https://arxiv.org/pdf/2406.06642) paper.

<p align="center">
  <img src="resources/workflow.jpg" width="700">
</p>

The main pipeline trains and evaluates a wide range of state-of-the-art TNNs and Graph Neural Networks (GNNs) (see <a href="#gear-neural-networks">:gear: Neural Networks</a>) on numerous and varied datasets and benchmark tasks (see <a href="#books-datasets">:books: Datasets</a> ). Additionally, the library offers the ability to transform, i.e. _lift_, each dataset from one topological domain to another (see <a href="#rocket-liftings">:rocket: Liftings</a>), enabling for the first time an exhaustive inter-domain comparison of TNNs.

## :jigsaw: Get Started

### Create Environment

If you do not have conda on your machine, please follow [their guide](https://docs.anaconda.com/free/miniconda/miniconda-install/) to install it. 

First, clone the `TopoBench` repository and set up a conda environment `tb` with python 3.11.3. 

```
git clone git@github.com:geometric-intelligence/topobench.git
cd TopoBench
conda create -n tb python=3.11.3
```

Next, check the CUDA version of your machine:
```
/usr/local/cuda/bin/nvcc --version
```
and ensure that it matches the CUDA version specified in the `env_setup.sh` file (`CUDA=cu121` by default). If it does not match, update `env_setup.sh` accordingly by changing both the `CUDA` and `TORCH` environment variables to compatible values as specified on [this website](https://github.com/pyg-team/pyg-lib).

Next, set up the environment with the following command.

```
source env_setup.sh
```
This command installs the `TopoBench` library and its dependencies. 

### Run Training Pipeline

Next, train the neural networks by running the following command:

```
python -m topobench 
```

Thanks to `hydra` implementation, one can easily override the default experiment configuration through the command line. For instance, the model and dataset can be selected as:

```
python -m topobench model=cell/cwn dataset=graph/MUTAG
```

**Remark:** By default, our pipeline identifies the source and destination topological domains, and applies a default lifting between them if required.

The same CLI override mechanism also applies when modifying more finer configurations within a `CONFIG GROUP`. Please, refer to the official [`hydra`documentation](https://hydra.cc/docs/intro/) for further details.



## :bike: Experiments Reproducibility
To reproduce Table 1 from the [`TopoBench: A Framework for Benchmarking Topological Deep Learning`](https://arxiv.org/pdf/2406.06642) paper, please run the following command:

```
bash scripts/reproduce.sh
```
**Remark:** We have additionally provided a public [W&B (Weights & Biases) project](https://wandb.ai/telyatnikov_sap/TopoBenchmark_main?nw=nwusertelyatnikov_sap) with logs for the corresponding runs (updated on June 11, 2024).


## :anchor: Tutorials

Explore our [tutorials](https://github.com/geometric-intelligence/TopoBench/tree/main/tutorials) for further details on how to add new datasets, transforms/liftings, and benchmark tasks. 

## :gear: Neural Networks

We list the neural networks trained and evaluated by `TopoBench`, organized by the topological domain over which they operate: graph, simplicial complex, cellular complex or hypergraph. Many of these neural networks were originally implemented in [`TopoModelX`](https://github.com/pyt-team/TopoModelX).


### Graphs
| Model | Reference |
| --- | --- |
| GAT | [Graph Attention Networks](https://openreview.net/pdf?id=rJXMpikCZ) |
| GIN | [How Powerful are Graph Neural Networks?](https://openreview.net/pdf?id=ryGs6iA5Km) |
| GCN | [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/pdf/1609.02907v4) |
| GraphMLP | [Graph-MLP: Node Classification without Message Passing in Graph](https://arxiv.org/pdf/2106.04051) |

### Simplicial complexes
| Model | Reference |
| --- | --- |
| SAN | [Simplicial Attention Neural Networks](https://arxiv.org/pdf/2203.07485) |
| SCCN | [Efficient Representation Learning for Higher-Order Data with Simplicial Complexes](https://openreview.net/pdf?id=nGqJY4DODN) |
| SCCNN | [Convolutional Learning on Simplicial Complexes](https://arxiv.org/pdf/2301.11163) |
| SCN | [Simplicial Complex Neural Networks](https://ieeexplore.ieee.org/document/10285604) |

### Cellular complexes
| Model | Reference |
| --- | --- |
| CAN | [Cell Attention Network](https://arxiv.org/pdf/2209.08179) |
| CCCN | Inspired by [A learning algorithm for computational connected cellular network](https://ieeexplore.ieee.org/document/1202221), implementation adapted from [Generalized Simplicial Attention Neural Networks](https://arxiv.org/abs/2309.02138)|
| CXN | [Cell Complex Neural Networks](https://openreview.net/pdf?id=6Tq18ySFpGU) |
| CWN | [Weisfeiler and Lehman Go Cellular: CW Networks](https://arxiv.org/pdf/2106.12575) |

### Hypergraphs
| Model | Reference |
| --- | --- |
| AllDeepSet | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/pdf?id=hpBTIv2uy_E) |
| AllSetTransformer | [You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks](https://openreview.net/pdf?id=hpBTIv2uy_E) |
| EDGNN | [Equivariant Hypergraph Diffusion Neural Operators](https://arxiv.org/pdf/2207.06680) |
| UniGNN | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956) |
| UniGNN2 | [UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks](https://arxiv.org/pdf/2105.00956) |

### Combinatorial complexes
| Model | Reference |
| --- | --- |
| GCCN | [TopoTune: A Framework for Generalized Combinatorial Complex Neural Networks](https://arxiv.org/pdf/2410.06530) |

**Remark:** TopoBench includes [TopoTune](https://arxiv.org/pdf/2410.06530), a comprehensive framework for easily defining and training new, general TDL models on any domain using any (graph) neural network as a backbone. Please check out our extended [TopoTune wiki page](https://github.com/geometric-intelligence/TopoBench/wiki/TopoTune) for further details on how to use this framework.


## :rocket: Liftings

We list the liftings used in `TopoBench` to transform datasets. Here, a _lifting_ refers to a function that transforms a dataset defined on a topological domain (_e.g._, on a graph) into the same dataset but supported on a different topological domain (_e.g._, on a simplicial complex).

<details>
<summary><b> Topology Liftings </b></summary>

### Graph2Simplicial
| Name | Description | Reference |
| --- | --- | --- |
| CliqueLifting | The algorithm finds the cliques in the graph and creates simplices. Given a clique the first simplex added is the one containing all the nodes of the clique, then the simplices composed of all the possible combinations with one node missing, then two nodes missing, and so on, until all the possible pairs are added. Then the method moves to the next clique. | [Simplicial Complexes](https://en.wikipedia.org/wiki/Clique_complex) |
| KHopLifting | For each node in the graph, take the set of its neighbors, up to k distance, and the node itself. These sets are then treated as simplices. The dimension of each simplex depends on the degree of the nodes. For example, a node with d neighbors forms a d-simplex. | [Neighborhood Complexes](https://arxiv.org/pdf/math/0512077) |

### Graph2Cell
| Name | Description | Reference |
| --- | --- | --- |
| CellCycleLifting |To lift a graph to a cell complex (CC) we proceed as follows. First, we identify a finite set of cycles (closed loops) within the graph. Second, each identified cycle in the graph is associated to a 2-cell, such that the boundary of the 2-cell is the cycle. The nodes and edges of the cell complex are inherited from the graph. | [Appendix B](https://arxiv.org/abs/2206.00606) |

### Graph2Hypergraph
| Name | Description | Reference |
| --- | --- | --- |
| KHopLifting | For each node in the graph, the algorithm finds the set of nodes that are at most k connections away from the initial node. This set is then used to create an hyperedge. The process is repeated for all nodes in the graph. | [Section 3.4](https://ieeexplore.ieee.org/abstract/document/9264674) |
| KNearestNeighborsLifting | For each node in the graph, the method finds the k nearest nodes by using the Euclidean distance between the vectors of features. The set of k nodes found is considered as an hyperedge. The proces is repeated for all nodes in the graph. | [Section 3.1](https://ieeexplore.ieee.org/abstract/document/9264674) |
</details>

<details>
  <summary><b> Feature Liftings <b></summary>

| Name                | Description                                                                 | Supported Domains |
|---------------------|-----------------------------------------------------------------------------|-------------------|
| ProjectionSum       | Projects r-cell features of a graph to r+1-cell structures utilizing incidence matrices \(B_{r}\). | Simplicial, Cell  |
| ConcatenationLifting | Concatenate r-cell features to obtain r+1-cell features.                   | Simplicial        |

</details>

<details>
  <summary><b> Data Transformations <b></summary>

| Transform | Description | Reference |
| --- | --- | --- |
| Message Passing Homophily | Higher-order homophily measure for hypergraphs | [Source](https://arxiv.org/abs/2310.07684) |
| Group Homophily | Higher-order homophily measure for hypergraphs that considers groups of predefined sizes  | [Source](https://arxiv.org/abs/2103.11818) |
</details>

## :books: Datasets

### Graphs
| Dataset | Task | Description | Reference |
| --- | --- | --- | --- |
| Cora | Classification | Cocitation dataset. | [Source](https://link.springer.com/article/10.1023/A:1009953814988) |
| Citeseer | Classification | Cocitation dataset. | [Source](https://dl.acm.org/doi/10.1145/276675.276685) |
| Pubmed | Classification | Cocitation dataset. | [Source](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2157) |
| MUTAG | Classification | Graph-level classification. | [Source](https://pubs.acs.org/doi/abs/10.1021/jm00106a046) |
| PROTEINS | Classification | Graph-level classification. | [Source](https://academic.oup.com/bioinformatics/article/21/suppl_1/i47/202991) |
| NCI1 | Classification | Graph-level classification. | [Source](https://ieeexplore.ieee.org/document/4053093) |
| NCI109 | Classification | Graph-level classification. | [Source](https://arxiv.org/pdf/2007.08663) |
| IMDB-BIN | Classification | Graph-level classification. | [Source](https://dl.acm.org/doi/10.1145/2783258.2783417) |
| IMDB-MUL | Classification | Graph-level classification. | [Source](https://dl.acm.org/doi/10.1145/2783258.2783417) |
| REDDIT | Classification | Graph-level classification. | [Source](https://proceedings.neurips.cc/paper_files/paper/2017/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf) |
| Amazon | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/1205.6233) |
| Minesweeper | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/2302.11640) |
| Empire | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/2302.11640) |
| Tolokers | Classification | Heterophilic dataset. | [Source](https://arxiv.org/pdf/2302.11640) |
| US-county-demos | Regression | In turn each node attribute is used as the target label. | [Source](https://arxiv.org/pdf/2002.08274) |
| ZINC | Regression | Graph-level regression. | [Source](https://pubs.acs.org/doi/10.1021/ci3001277) |

### Hypergraphs
| Dataset | Task | Description | Reference |
| --- | --- | --- | --- |
| Cora-Cocitation | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| Citeseer-Cocitation | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| PubMed-Cocitation | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| Cora-Coauthorship | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |
| DBLP-Coauthorship | Classification | Cocitation dataset. | [Source](https://proceedings.neurips.cc/paper_files/paper/2019/file/1efa39bcaec6f3900149160693694536-Paper.pdf) |



## :mag: References ##

To learn more about `TopoBench`, we invite you to read the paper:

```
@article{telyatnikov2024topobench,
      title={TopoBench: A Framework for Benchmarking Topological Deep Learning}, 
      author={Lev Telyatnikov and Guillermo Bernardez and Marco Montagna and Pavlo Vasylenko and Ghada Zamzmi and Mustafa Hajij and Michael T Schaub and Nina Miolane and Simone Scardapane and Theodore Papamarkou},
      year={2024},
      eprint={2406.06642},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.06642}, 
}
```
If you find `TopoBench` useful, we would appreciate if you cite us!



## :mouse: Additional Details
<details>
<summary><b>Hierarchy of configuration files</b></summary>

```
├── configs                   <- Hydra configs
│   ├── callbacks                <- Callbacks configs
│   ├── dataset                  <- Dataset configs
│   │   ├── graph                    <- Graph dataset configs
│   │   ├── hypergraph               <- Hypergraph dataset configs
│   │   └── simplicial               <- Simplicial dataset configs
│   ├── debug                    <- Debugging configs
│   ├── evaluator                <- Evaluator configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── loss                     <- Loss function configs
│   ├── model                    <- Model configs
│   │   ├── cell                     <- Cell model configs
│   │   ├── graph                    <- Graph model configs
│   │   ├── hypergraph               <- Hypergraph model configs
│   │   └── simplicial               <- Simplicial model configs
│   ├── optimizer                <- Optimizer configs
│   ├── paths                    <- Project paths configs
│   ├── scheduler                <- Scheduler configs
│   ├── trainer                  <- Trainer configs
│   ├── transforms               <- Data transformation configs
│   │   ├── data_manipulations       <- Data manipulation transforms
│   │   ├── dataset_defaults         <- Default dataset transforms
│   │   ├── feature_liftings         <- Feature lifting transforms
│   │   └── liftings                 <- Lifting transforms
│   │       ├── graph2cell               <- Graph to cell lifting transforms
│   │       ├── graph2hypergraph         <- Graph to hypergraph lifting transforms
│   │       ├── graph2simplicial         <- Graph to simplicial lifting transforms
│   │       ├── graph2cell_default.yaml  <- Default graph to cell lifting config
│   │       ├── graph2hypergraph_default.yaml <- Default graph to hypergraph lifting config
│   │       ├── graph2simplicial_default.yaml <- Default graph to simplicial lifting config
│   │       ├── no_lifting.yaml           <- No lifting config
│   │       ├── custom_example.yaml       <- Custom example transform config
│   │       └── no_transform.yaml         <- No transform config
│   ├── wandb_sweep              <- Weights & Biases sweep configs
│   │
│   ├── __init__.py              <- Init file for configs module
│   └── run.yaml               <- Main config for training
```


</details>

<details>
<summary><b> More information regarding Topological Deep Learning </b></summary>
  
  [Topological Graph Signal Compression](https://arxiv.org/pdf/2308.11068)
  
  [Architectures of Topological Deep Learning: A Survey on Topological Neural Networks](https://par.nsf.gov/servlets/purl/10477141)
  
  [TopoX: a suite of Python packages for machine learning on topological domains](https://arxiv.org/pdf/2402.02441)	
</details>
