# Grax: Graph Neural Networks in Jax

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project aims to provide re-implementations of neural networks in [jax][jax], as close to the original author's implementations as practical. Known deviations from original implementation logic are documented.

This project uses the following large open-source projects:

- [jax][jax] for performant low-level operations;
- [haiku][haiku] for parameter / state management;
- [optax][optax] for optimizer implementations; and
- [tensorflow-datasets](https://github.com/tensorflow/datasets) as a dataset framework.

Additional functionality is provided in smaller repositories:

- [graph-tfds](https://github.com/jackd/graph-tfds) for dataset implementations;
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

#### [DGL](https://github.com/jackd/graph-tfds): Citations

Citations datasets use [dgl](https://github.com/dmlc/dgl) which will be installed with the above. You can customize where to download/extract relevant files with:

```bash
export DGL_DATA=/path/to/dgl_data_dir  # otherwise uses ~/.dgl
```

#### [Open Graph Benchmark](https://ogb.stanford.edu/)

```bash
pip install ogb
export OGB_DATA=/path/to/ogb_data_dir  # otherwise uses ~/ogb
```

#### [graph-tfds](https://github.com/jackd/graph-tfds): Amazon and Coauthor

`amazon/computers`, `amazon/photo`, `coauthor/cs` and `coauthor/physics` require [graph-tfds](https://github.com/jackd/graph-tfds) (the `dgl` versions do not contain train/validation/test splits).

```bash
pip install tensorflow  # or tensorflow-cpu, tf-nightly etc
git clone https://github.com/jackd/graph-tfds
cd graph-tfds
pip install -r requirements.txt
pip install -e .
export TFDS_DATA=/path/to/tfds_data_dir  # otherwise uses ~/tensorflow_datasets
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
