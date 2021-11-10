import os
import typing as tp
from dataclasses import dataclass
from functools import partial

import dgl
import gin
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import spax
from huf.types import PRNGKey
from jax.experimental.sparse.ops import COO, JAXSparse
from spax import ops

from grax.graph_utils import algorithms
from grax.huf_utils import SplitData
from grax.problems.single.splits import split_by_class

configurable = partial(gin.configurable, module="grax.problems.single")

T = tp.TypeVar("T")


def ids_to_mask(ids: jnp.ndarray, size: int, dtype=bool):
    assert ids.ndim == 1
    return jnp.zeros((size,), dtype).at[ids].set(jnp.ones((ids.size,), dtype))


@dataclass
class SemiSupervisedSingle:
    """Data class for a single sparsely labelled graph."""

    # Node features might be np.ndarrays to avoid excessive memory requirements
    node_features: tp.Union[jnp.ndarray, JAXSparse, np.ndarray]  # [N, F]
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
        return self.graph.data.size // 2

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


def save_h5(path: str, data: SemiSupervisedSingle):
    assert not os.path.isfile(path)
    graph = data.graph
    assert isinstance(graph, COO)
    assert jnp.all(graph.data == 1)
    with h5py.File(path, "w") as fp:
        for k, v in (
            ("node_features", data.node_features),
            ("row", graph.row),
            ("col", graph.col),
            ("labels", data.labels),
            ("train_ids", data.train_ids),
            ("validation_ids", data.validation_ids),
            ("test_ids", data.test_ids),
        ):
            fp.create_dataset(k, data=np.asarray(v))
        fp.attrs["num_classes"] = data.num_classes


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
def remove_back_edges(single: SemiSupervisedSingle) -> SemiSupervisedSingle:
    nn = single.num_nodes
    lil = [[] for _ in range(nn)]
    for r, c in zip(single.graph.row.to_py(), single.graph.col.to_py()):
        # lil[r].append(c)
        lil[c].append(r)
    algorithms.remove_back_edges(lil)
    lengths = jnp.array([len(l) for l in lil])
    rows = jnp.repeat(jnp.arange(nn, dtype=jnp.int32), lengths)
    cols = jnp.concatenate(
        [np.array(l, np.int32) if len(l) else np.zeros((0,), np.int32) for l in lil],
        axis=0,
    )
    graph = COO((jnp.ones((rows.size,)), rows, cols), shape=(nn, nn))
    return SemiSupervisedSingle(
        single.node_features,
        graph,
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
    graph: COO, dtype: jnp.dtype = jnp.int32, directed: bool = True, connection="weak"
) -> jnp.ndarray:
    # using scipy.sparse.csgraph.connected_components
    graph = spax.ops.to_csr(graph)
    graph = sp.csr_matrix((graph.data, graph.indices, graph.indptr), shape=graph.shape)
    ncomponents, labels = sp.csgraph.connected_components(
        graph, return_labels=True, directed=directed, connection=connection
    )
    if ncomponents == 1:
        return jnp.arange(graph.shape[0], dtype=jnp.dtype)
    sizes = [np.count_nonzero(labels == i) for i in range(ncomponents)]
    i = np.argmax(sizes)
    (indices,) = np.where(labels == i)
    return jnp.asarray(indices, dtype)


# def get_largest_component_indices(
#     graph: COO, dtype: jnp.dtype = jnp.int32
# ) -> jnp.ndarray:
#     # create nx graph
#     g = nx.Graph()
#     graph = spax.ops.to_coo(graph)
#     coords = jnp.stack((graph.row, graph.col), axis=1)
#     for u, v in coords:
#         g.add_edge(u, v)
#     if nx.is_connected(g):
#         return jnp.arange(graph.row.size, dtype=dtype)
#     # enumerate used for tie-breaking purposes
#     _, _, component = max(
#         ((len(c), i, c) for i, c in enumerate(nx.connected_components(g)))
#     )
#     return jnp.asarray(sorted(component), dtype=dtype)


@configurable
def get_largest_component(single: SemiSupervisedSingle) -> SemiSupervisedSingle:
    indices = get_largest_component_indices(single.graph)
    return subgraph(single, indices)


def _load_dgl_graph(dgl_example, make_symmetric=False):
    r, c = (x.numpy() for x in dgl_example.edges())
    shape = (dgl_example.num_nodes(),) * 2
    if make_symmetric:
        # add symmetric edges
        r = np.array(r, dtype=np.int64)
        c = np.array(c, dtype=np.int64)
        # remove diagonals
        valid = r != c
        r = r[valid]
        c = c[valid]
        r, c = np.concatenate((r, c)), np.concatenate((c, r))
        i1d = np.ravel_multi_index((r, c), shape)
        i1d = np.unique(i1d)  # also sorts
        r, c = np.unravel_index(  # pylint: disable=unbalanced-tuple-unpacking
            i1d, shape
        )
    # return sp.coo_matrix((np.ones((r.size,), dtype=np.float32), (r, c)), shape=shape)
    return COO((jnp.ones((r.size,), dtype=jnp.float32), r, c), shape=shape)


def _load_dgl_example(
    dgl_example, make_symmetric=False, sparse_features=False
) -> SemiSupervisedSingle:
    feat, label = (dgl_example.ndata[k].numpy() for k in ("feat", "label"))
    if sparse_features:
        i, j = np.where(feat)
        feat = COO((feat[i, j], i, j), shape=feat.shape)
    train_ids, validation_ids, test_ids = (
        jnp.where(dgl_example.ndata[k].numpy())[0] if k in dgl_example.ndata else None
        for k in ("train_mask", "val_mask", "test_mask")
    )
    graph = _load_dgl_graph(dgl_example, make_symmetric=make_symmetric)
    label = jnp.asarray(label)

    return SemiSupervisedSingle(feat, graph, label, train_ids, validation_ids, test_ids)


def _get_dir(data_dir: tp.Optional[str], environ: str, default: tp.Optional[str]):
    if data_dir is None:
        data_dir = os.environ.get(environ, default)
    if data_dir is None:
        return None
    return os.path.expanduser(os.path.expandvars(data_dir))


_dgl_constructors = {
    "cora": dgl.data.CoraGraphDataset,
    "pubmed": dgl.data.PubmedGraphDataset,
    "citeseer": dgl.data.CiteseerGraphDataset,
    "amazon/computer": dgl.data.AmazonCoBuyComputerDataset,
    "amazon/photo": dgl.data.AmazonCoBuyPhotoDataset,
    "coauthor/physics": dgl.data.CoauthorPhysicsDataset,
    "coauthor/cs": dgl.data.CoauthorCSDataset,
}


@configurable
def get_data(name: str, **kwargs):
    if name in _dgl_constructors:
        return dgl_data(name, **kwargs)
    if name.startswith("ogbn-"):
        return ogbn_data(name[5:], **kwargs)
    raise ValueError(f"Unrecognized name {name}")


@configurable
def dgl_data(
    name: str,
    data_dir: tp.Optional[str] = None,
    make_symmetric: bool = True,
    sparse_features: bool = False,
) -> SemiSupervisedSingle:
    raw_dir = _get_dir(data_dir, "DGL_DATA", None)
    ds = _dgl_constructors[name](raw_dir=raw_dir)
    return _load_dgl_example(
        ds[0], make_symmetric=make_symmetric, sparse_features=sparse_features
    )


@configurable
def ogbn_data(
    name: str, data_dir: tp.Optional[str] = None, make_symmetric: bool = True,
):
    import ogb.nodeproppred  # pylint: disable=import-outside-toplevel

    root_dir = _get_dir(data_dir, "OGB_DATA", "~/ogb")

    print(f"Loading dgl ogbn-{name}...")
    ds = ogb.nodeproppred.DglNodePropPredDataset(f"ogbn-{name}", root=root_dir)
    print("Got base data. Initial preprocessing...")
    split_ids = ds.get_idx_split()
    train_ids, validation_ids, test_ids = (
        jnp.asarray(split_ids[n].numpy()) for n in ("train", "valid", "test")
    )
    example, labels = ds[0]
    feats = example.ndata["feat"].numpy()
    labels = labels.numpy().squeeze(1)
    labels[np.isnan(labels)] = -1
    labels = jnp.asarray(labels.astype(np.int32))
    graph = _load_dgl_graph(example, make_symmetric=make_symmetric)
    print("Finished initial preprocessing")

    data = SemiSupervisedSingle(
        feats, graph, labels, train_ids, validation_ids, test_ids
    )
    print("num (nodes, edges, features):")
    print(data.num_nodes, data.num_edges, data.node_features.shape[1])
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
) -> SemiSupervisedSingle:
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
    data: SemiSupervisedSingle,
    *,
    largest_component: bool = False,
    with_back_edges: bool = True,
    id_transform: tp.Optional[SingleTransform] = None,
    graph_transform: tp.Union[GraphTransform, tp.Iterable[GraphTransform]] = (),
    node_features_transform: tp.Union[
        FeatureTransform, tp.Iterable[FeatureTransform]
    ] = (),
    transform: tp.Union[SingleTransform, tp.Iterable[SingleTransform]] = (),
    as_split: bool = True,
):
    """
    Applies common transforms sequentially.

    - `get_largest_component` if `largest_transform is True`
    - id_transform if not None
    - `remove_back_edges` if not with_back_edges
    - graph_transform
    - node_features_transform
    - transform
    """
    with jax.experimental.enable_x64():
        if largest_component:
            data = get_largest_component(data)
        if id_transform is not None:
            data = id_transform(data)
        if not with_back_edges:
            data = remove_back_edges(data)
        for t in as_iterable(graph_transform):
            data = apply_graph_transform(data, t)
        for t in as_iterable(node_features_transform):
            data = apply_node_features_transform(data, t)
        for t in as_iterable(transform):
            data = t(data)
        if as_split:
            data = as_single_example_splits(data)

    return data


@configurable
def as_single_example_splits(data: SemiSupervisedSingle) -> SplitData:
    features = data.node_features
    if isinstance(features, np.ndarray):
        features = jnp.asarray(features)
    train_ex, validation_ex, test_ex = (
        ((data.graph, features), data.labels, ids_to_mask(ids, data.num_nodes),)
        for ids in (data.train_ids, data.validation_ids, data.test_ids)
    )
    return SplitData((train_ex,), (validation_ex,), (test_ex,))
