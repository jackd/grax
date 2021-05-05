import typing as tp
from dataclasses import dataclass
from functools import partial

import gin
import networkx as nx
import tensorflow as tf
import tensorflow_datasets as tfds

import jax
import jax.numpy as jnp
from graph_tfds.graphs import (  # pylint: disable=unused-import
    amazon,
    cite_seer,
    coauthor,
    cora,
    pub_med,
)
from grax.graph_utils import laplacians, transforms
from grax.problems.single.splits import split_by_class
from huf.types import PRNGKey
from jax.experimental.sparse_ops import COO, JAXSparse
from spax import ops

configurable = partial(gin.configurable, module="grax.problems.single")

T = tp.TypeVar("T")


def ids_to_mask(ids: jnp.ndarray, size: int, dtype=bool):
    assert ids.ndim == 1
    return jnp.zeros((size,), dtype).at[ids].set(jnp.ones((ids.size,), dtype))


@dataclass
class SemiSupervisedSingle:
    """Data class for a single sparsely labelled graph."""

    node_features: tp.Union[
        jnp.ndarray, JAXSparse,
    ]  # [N, F]
    graph: JAXSparse  # [N, N]
    labels: jnp.ndarray  # [N]
    train_ids: tp.Optional[jnp.ndarray]  # [n_train << N]
    validation_ids: tp.Optional[jnp.ndarray]  # [n_eval < N]
    test_ids: tp.Optional[jnp.ndarray]  # [n_test < N]

    def __post_init__(self):
        for ids in (self.train_ids, self.validation_ids, self.test_ids):
            assert ids is None or ids.ndim == 1, ids.shape
            assert ids is None or jnp.issubdtype(ids.dtype, jnp.integer), ids.dtype

    @property
    def num_nodes(self) -> int:
        return self.graph.shape[0]

    @property
    def num_edges(self) -> int:
        return self.graph.nnz // 2

    def as_dict(self):
        return dict(
            node_features=self.node_features,
            graph=self.graph,
            labels=self.labels,
            train_ids=self.train_ids,
            validation_ids=self.validation_ids,
            test_ids=self.test_ids,
        )

    def rebuild(self, **updates):
        d = self.as_dict()
        d.update(updates)
        return SemiSupervisedSingle(**d)

    @property
    def train_mask(self) -> jnp.ndarray:
        return ids_to_mask(self.train_ids, self.num_nodes)

    @property
    def validation_mask(self) -> jnp.ndarray:
        return ids_to_mask(self.validation_ids, self.num_nodes)

    @property
    def test_mask(self) -> jnp.ndarray:
        return ids_to_mask(self.test_ids, self.num_nodes)

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1


@configurable
def symmetric_normalize_with_row_sum(single: SemiSupervisedSingle):
    rs = ops.norm(single.graph, axis=1, ord=1)
    factor = jnp.where(rs == 0, jnp.zeros_like(rs), 1.0 / jnp.sqrt(rs))
    graph = ops.scale_columns(ops.scale_rows(single.graph, factor), factor)
    return SemiSupervisedSingle(
        (rs, single.node_features),
        graph,
        single.labels,
        single.train_ids,
        single.validation_ids,
        single.test_ids,
    )


@configurable
def symmetric_normalize_laplacian_with_row_sum(
    single: SemiSupervisedSingle, shift: float = 0.0, scale: float = 1.0
):
    laplacian, rs = laplacians.normalized_laplacian(single.graph)
    return SemiSupervisedSingle(
        (rs, single.node_features),
        transforms.linear_transform(laplacian, shift=shift, scale=scale),
        single.labels,
        single.train_ids,
        single.validation_ids,
        single.test_ids,
    )


def subgraph(
    single: SemiSupervisedSingle, indices: jnp.ndarray
) -> SemiSupervisedSingle:
    indices = jnp.asarray(indices, jnp.int32)
    assert indices.ndim == 1, indices.shape
    index_dtype = indices.dtype
    assert jnp.issubdtype(index_dtype, jnp.integer)
    indices = jnp.sort(indices)
    adj = ops.gather(ops.gather(single.graph, indices, axis=0), indices, axis=1)

    node_features = single.node_features
    if isinstance(node_features, JAXSparse):
        node_features = ops.gather(node_features, indices, axis=0)
    else:
        node_features = node_features[indices]

    remap_indices = (
        jnp.zeros((single.graph.shape[0],), index_dtype)
        .at[indices]
        .set(jnp.arange(indices.size, dtype=index_dtype))
    )

    def valid_ids(ids):
        return None if ids is None else remap_indices[ids]

    return SemiSupervisedSingle(
        node_features,
        adj,
        single.labels[indices],
        train_ids=valid_ids(single.train_ids),
        validation_ids=valid_ids(single.validation_ids),
        test_ids=valid_ids(single.test_ids),
    )


def get_largest_component_indices(
    graph: COO, dtype: jnp.dtype = jnp.int32
) -> SemiSupervisedSingle:
    # create nx graph
    g = nx.Graph()
    for u, v in graph.tocoo().coords.T:
        g.add_edge(u, v)
    if nx.is_connected(g):
        return jnp.arange(graph.coords.shape[1], dtype=dtype)
    # enumerate used for tie-breaking purposes
    _, _, component = max(
        ((len(c), i, c) for i, c in enumerate(nx.connected_components(g)))
    )
    return jnp.asarray(sorted(component), dtype=dtype)


@configurable
def get_largest_component(single: SemiSupervisedSingle):
    return subgraph(single, get_largest_component_indices(single.graph))


@configurable
def citations_data(name: str = "cora") -> SemiSupervisedSingle:
    """
    Get semi-supervised citations data.

    Args:
        name: one of "cora", "cite_seer", "pub_med", or a registered tfds builder name
            with the same element spec.

    Returns:
        `SemiSupervisedSingle`. The graph matrix has forward/back edges but no self
            edges and uniform weights of `1.0`.
    """
    dataset = tfds.load(name)
    if isinstance(dataset, dict):
        if len(dataset) == 1:
            (dataset,) = dataset.values()
        else:
            raise ValueError(
                f"tfds builder {name} had more than 1 split ({sorted(dataset.keys())})."
                " Please use 'name/split'"
            )
    element = tf.data.experimental.get_single_element(dataset)
    graph = element["graph"]
    features = graph["node_features"]
    row, col = graph["links"].numpy().T
    n = features.shape[0]
    graph = COO((jnp.ones(row.shape, dtype=jnp.float32), row, col), shape=(n, n))
    if isinstance(features, tf.SparseTensor):
        row, col = features.indices.numpy().T
        features = COO((features.values.numpy(), row, col), shape=features.shape)
    else:
        features = jnp.asarray(features.numpy())
    labels, train_ids, validation_ids, test_ids = (
        jnp.asarray(element[k])
        for k in ("node_labels", "train_ids", "validation_ids", "test_ids")
    )

    return SemiSupervisedSingle(
        features, graph, labels, train_ids, validation_ids, test_ids
    )


@configurable
def pitfalls_data(name: str = "amazon/computers"):
    dataset = tfds.load(name)
    if isinstance(dataset, dict):
        if len(dataset) == 1:
            (dataset,) = dataset.values()
        else:
            raise ValueError(
                f"tfds builder {name} had more than 1 split ({sorted(dataset.keys())})."
                " Please use 'name/split'"
            )
    element = tf.data.experimental.get_single_element(dataset)
    graph = element["graph"]
    features = graph["node_features"]
    num_nodes = int(features.shape[0])
    coords = graph["links"].numpy().T
    row, col = coords
    coords = coords[:, row != col]  # remove self-loops
    graph = COO(
        coords,
        jnp.ones((coords.shape[1],), dtype=jnp.float32),
        shape=(num_nodes, num_nodes),
    )
    # add reverse edges
    graph = ops.add(graph, ops.transpose(graph))
    labels = jnp.asarray(element["node_labels"].numpy())

    if isinstance(features, tf.SparseTensor):
        features = COO(
            features.indices.numpy().T, features.values.numpy(), tuple(features.shape),
        )
    else:
        features = jnp.asarray(features.numpy())

    data = SemiSupervisedSingle(features, graph, labels, None, None, None)
    return data


Transform = tp.Union[tp.Callable[[SemiSupervisedSingle], SemiSupervisedSingle], None]


def as_iterable(x) -> tp.Iterable:
    if hasattr(x, "__iter__"):
        return x
    return (x,)


@configurable
def transformed(
    base: SemiSupervisedSingle, transforms: tp.Union[Transform, tp.Iterable[Transform]],
) -> SemiSupervisedSingle:
    for transform in as_iterable(transforms):
        if transform is not None:
            base = transform(base)
    return base


def apply_node_features_transform(
    base: SemiSupervisedSingle, transform_fun: tp.Callable
):
    return base.rebuild(node_features=transform_fun(base.node_features))


@configurable
def node_features_transform(transform_fun: tp.Callable):
    return partial(apply_node_features_transform, transform_fun=transform_fun)


def apply_graph_transform(base: SemiSupervisedSingle, transform_fun: tp.Callable):
    return base.rebuild(graph=transform_fun(base.graph))


@configurable
def graph_transform(transform_fun: tp.Callable):
    return partial(apply_graph_transform, transform_fun=transform_fun)


SingleTransform = tp.Callable[[SemiSupervisedSingle], SemiSupervisedSingle]
GraphTransform = tp.Callable[[JAXSparse], JAXSparse]
FeatureTransform = tp.Callable[[jnp.ndarray], jnp.ndarray]


@configurable
def randomize_splits(
    single: SemiSupervisedSingle,
    train_samples_per_class: int,
    validation_samples_per_class: int,
    rng: tp.Union[PRNGKey, int] = 0,
):
    if isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)
    splits = split_by_class(
        rng,
        single.labels,
        (train_samples_per_class, validation_samples_per_class),
        single.num_classes,
    )
    return SemiSupervisedSingle(
        single.node_features, single.graph, single.labels, *splits
    )


@configurable
def transformed_simple(
    base: SemiSupervisedSingle,
    *,
    largest_component: bool = False,
    graph_transform: tp.Union[GraphTransform, tp.Iterable[GraphTransform]] = (),
    node_features_transform: tp.Union[
        FeatureTransform, tp.Iterable[FeatureTransform]
    ] = (),
    transform: tp.Union[SingleTransform, tp.Iterable[SingleTransform]] = (),
):
    """
    Applies common transforms sequentially.

    - `get_largest_component` if `largest_transform is True`
    - graph_transform
    - node_features_transform
    - transform
    """
    if largest_component:
        base = get_largest_component(base)
    for t in as_iterable(graph_transform):
        base = apply_graph_transform(base, t)
    for t in as_iterable(node_features_transform):
        base = apply_node_features_transform(base, t)
    for t in as_iterable(transform):
        base = t(base)
    return base
