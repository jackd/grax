# Grax: Graph Neural Networks in Jax

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project aims to provide re-implementations of neural networks in [jax][jax], as close to the original author's implementations as practical. Apart from different default initializations,known deviations from original implementation logic are documented.

Implementations are in their own [projects](grax/projects). Current implementations include:

- [APPNP](grax/projects/appnp): Approximate Personalized Propagation of Neural Predictions
- [DAGNN](grax/projects/dagnn): Deep Adaptive Graph Neural Networks
- [DEQ_GCN](grax/projects/deq_gcn): (Stalled WIP) Deep Equilibrium Graph Convolution Networks
- [GAT](grax/projects/gat): Graph Attention Networks
- [GCN](grax/projects/gcn): Graph Convolution Networks
- [GCN2](grax/projects/gcn2): Graph Convolution Networks 2
- [IGAT](grax/projects/igat): (Stalled WIP) Inverse Graph Attention Networks
- [igcn](graph/projects/igcn): Inverse Graph Convolution Networks
- [pigcn](graph/projects/pigcn): Pseudo-inverse Graph Convolution Networks
- [sgc](graph/projects/sgc): Simple Graph Convolution Networks

See the relevant subdirectory `README.md` for more details and example usage.

## Dependencies

This project uses the following large open-source projects:

- [jax][jax] for performant low-level operations;
- [haiku][haiku] for parameter / state management;
- [optax][optax] for optimizer implementations; and
- [gin][gin] for configuration.

Additional functionality is provided in smaller repositories:

- [spax](https://github.com/jackd/spax) for sparse [jax][jax] classes and operations; and
- [huf](https://github.com/jackd/huf): minimal framework built on top of [haiku][haiku] and [optax][optax].

This library is in early rapid development - things will break frequently.

## Installation

After installing [jax][jax],

```bash
git clone https://github.com/jackd/grax
cd grax
pip install -r requirements.txt
pip install -e .  # local install
```

### Datasets

#### [DGL](https://github.com/dmlc/dgl): Citations, Amazon and Coauthor

Citations datasets use [dgl](https://github.com/dmlc/dgl) which will be installed with the above. You can customize where to download/extract relevant files with:

```bash
export DGL_DATA=/path/to/dgl_data_dir  # otherwise uses ~/.dgl
```

#### [Open Graph Benchmark](https://ogb.stanford.edu/)

```bash
pip install ogb
export OGB_DATA=/path/to/ogb_data_dir  # otherwise uses ~/ogb
```

## Quick Start

After installing:

```bash
# run a single GCN model on pubmed dataset
python -m grax grax_config/single/fit.gin gcn/config/pubmed.gin
# customize configuration
python -m grax grax_config/single/fit.gin gcn/config/pubmed.gin --bindings='
dropout_rate=0.6
seed=1
'
# perform multiple runs
python -m grax grax_config/single/fit_many.gin gcn/config/pubmed.gin
# perform multiple runs with ray
python -m grax grax_config/single/ray/fit_many.gin gcn/config/pubmed.gin
```

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```

[jax]: https://github.com/google/jax
[haiku]: https://github.com/deepmind/dm-haiku
[optax]: https://github.com/deepmind/optax
[gin]: https://github.com/google/gin
