import typing as tp
from dataclasses import dataclass
from functools import partial

import gin
import jax.numpy as jnp
import networkx as nx
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.graphs import cite_seer, cora, pub_med  # pylint: disable=unused-import
from spax import COO, SparseArray, ops

configurable = partial(gin.configurable, module="grax.problems.single")

T = tp.TypeVar("T")


def ids_to_mask(ids: jnp.ndarray, size: int, dtype=bool):
    assert ids.ndim == 1
    return jnp.zeros((size,), dtype).at[ids].set(jnp.ones((ids.size,), dtype))


@dataclass
class SemiSupervisedSingle:
    """Data class for a single sparsely labelled graph."""

    node_features: tp.Union[jnp.ndarray, SparseArray]  # [N, F]
    graph: SparseArray  # [N, N]
    labels: jnp.ndarray  # [N]
    train_ids: jnp.ndarray  # [n_train << N]
    validation_ids: jnp.ndarray  # [n_eval < N]
    test_ids: jnp.ndarray  # [n_test < N]

    def __post_init__(self):
        assert self.node_features.ndim == 2, self.node_features.shape
        assert jnp.issubdtype(
            self.node_features.dtype, jnp.floating
        ), self.node_features.dtype

        for ids in (self.train_ids, self.validation_ids, self.test_ids):
            assert ids.ndim == 1, ids.shape
            assert jnp.issubdtype(ids.dtype, jnp.integer), ids.dtype

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

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


def subgraph(single: SemiSupervisedSingle, indices: jnp.ndarray):
    indices = jnp.asarray(indices, jnp.int32)
    assert indices.ndim == 1, indices.shape
    index_dtype = indices.dtype
    assert jnp.issubdtype(index_dtype, jnp.integer)
    indices = jnp.sort(indices)
    adj = ops.gather(ops.gather(single.graph, indices, axis=0), indices, axis=1)

    node_features = single.node_features
    if isinstance(node_features, SparseArray):
        node_features = ops.gather(node_features, indices, axis=0)
    else:
        node_features = node_features[indices]

    remap_indices = (
        jnp.zeros((single.graph.shape[0],), index_dtype)
        .at[indices]
        .set(jnp.arange(indices.size, dtype=index_dtype))
    )

    def valid_ids(ids):
        return remap_indices[ids]

    return SemiSupervisedSingle(
        node_features,
        adj,
        single.labels[indices],
        train_ids=valid_ids(single.train_ids),
        validation_ids=valid_ids(single.validation_ids),
        test_ids=valid_ids(single.test_ids),
    )


@configurable
def get_largest_component(single: SemiSupervisedSingle):
    # create nx graph
    g = nx.Graph()
    for u, v in single.graph.tocoo().coords.T:
        g.add_edge(u, v)
    if nx.is_connected(g):
        return single

    # enumerate used for tie-breaking purposes
    _, _, component = max(
        ((len(c), i, c) for i, c in enumerate(nx.connected_components(g)))
    )
    return subgraph(single, tuple(component))


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
    coords = graph["links"].numpy().T
    graph = COO(coords, jnp.ones((coords.shape[1],), dtype=jnp.float32))
    if isinstance(features, tf.SparseTensor):
        features = COO(
            features.indices.numpy().T, features.values.numpy(), tuple(features.shape)
        )
    else:
        features = jnp.asarray(features.numpy())
    labels, train_ids, validation_ids, test_ids = (
        jnp.asarray(element[k])
        for k in ("node_labels", "train_ids", "validation_ids", "test_ids")
    )

    return SemiSupervisedSingle(
        features, graph, labels, train_ids, validation_ids, test_ids
    )


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
GraphTransform = tp.Callable[[SparseArray], SparseArray]
FeatureTransform = tp.Callable[[jnp.ndarray], jnp.ndarray]


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
