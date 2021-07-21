import abc
import os
import typing as tp
from functools import partial

import gin
import h5py
import huf.data
import jax
import jax.numpy as jnp
import scipy.sparse as sp
import spax
import tqdm
from haiku import PRNGSequence
from jax.experimental.sparse.ops import COO
from spax.linalg import linear_operators as lin_ops

from grax.graph_utils.transforms import from_scipy, to_scipy
from grax.huf_utils import SplitData
from grax.problems.single.data import SemiSupervisedSingle

# from grax.graph_utils.transforms import symmetric_normalize

configurable = partial(gin.configurable, module="igcn.data")


def ids_to_mask_matrix(ids: jnp.ndarray, size: int, dtype=jnp.float32):
    nl = ids.size
    return (
        jnp.zeros((size, nl), dtype=dtype)
        .at[ids, jnp.arange(nl, dtype=ids.dtype)]
        .set(jnp.ones((nl,), dtype=dtype))
    )


def _get_propagator(
    adj: COO,
    *,
    epsilon: float = 0.1,
    tol: float = 1e-5,
    maxiter: tp.Optional[int] = None,
    rescale: bool = False,
) -> lin_ops.LinearOperator:
    assert maxiter is None or isinstance(maxiter, int)
    if epsilon == 1:
        return lin_ops.Identity(n=adj.shape[0], dtype=jnp.float32)

    x = adj

    # x = spax.ops.scale(symmetric_normalize(x), -(1 - epsilon))
    # # x = lin_ops.identity_plus(lin_ops.MatrixWrapper(x, is_self_adjoint=True))
    # with jax.experimental.enable_x64():
    #     x = spax.ops.add(spax.eye(x.shape[0], x.dtype, x.row.dtype), x)

    x = to_scipy(x)
    assert sp.isspmatrix_coo(x)
    factor = 1 / (sp.linalg.norm(x, ord=1, axis=1) ** 0.5)
    x = sp.coo_matrix(
        (x.data * factor[x.row] * factor[x.col], (x.row, x.col)), shape=x.shape
    )
    x = sp.eye(x.shape[0], dtype=x.dtype) - (1 - epsilon) * x
    x = x.tocoo()
    x = from_scipy(x)

    x = lin_ops.scatter_limit_split(x, is_self_adjoint=True)
    x = lin_ops.SelfAdjointInverse(x, tol=tol, maxiter=maxiter)
    if rescale:
        x = lin_ops.scale(x, epsilon)
    return x


@configurable
def preprocessed_logit_propagated_data(
    propagator: lin_ops.LinearOperator,
    features: tp.Union[COO, jnp.ndarray],
    labels: jnp.ndarray,
    ids: jnp.ndarray,
):
    prop = (propagator @ ids_to_mask_matrix(ids, propagator.shape[1])).T
    return (((prop, features, ids), labels[ids]),)


@configurable
def lazy_logit_propagated_data(
    propagator: lin_ops.LinearOperator,
    features: tp.Union[COO, jnp.ndarray],
    labels: jnp.ndarray,
    ids: jnp.ndarray,
):
    # return (((lin_ops.take(propagator, ids), features), labels[ids]),)
    return (
        (
            (lin_ops.take(propagator, ids), features, ids),
            labels[ids],
            # ids_to_mask(ids, propagator.shape[0], dtype=jnp.float32),
        ),
    )


@configurable
def input_propagated_data(
    propagator: lin_ops.LinearOperator,
    features: tp.Union[COO, jnp.ndarray],
    labels: jnp.ndarray,
    ids: jnp.ndarray,
    concat_original: bool = True,
):
    dense_features = spax.ops.to_dense(features)
    prop_features = (propagator @ dense_features)[ids]
    dense_features = dense_features[ids]
    if concat_original:
        if isinstance(features, COO):
            features = lin_ops.HStacked(spax.ops.to_coo(dense_features), prop_features)
        else:
            features = jnp.concatenate((dense_features, prop_features), axis=-1)
    else:
        features = prop_features
    return ((features, labels[ids]),)


@configurable
def get_split_data(
    data: SemiSupervisedSingle,
    epsilon: float,
    tol: float = 1e-5,
    rescale: bool = False,
    train_fun: tp.Callable = preprocessed_logit_propagated_data,
    validation_fun: tp.Callable = preprocessed_logit_propagated_data,
    test_fun: tp.Callable = lazy_logit_propagated_data,
) -> SplitData:
    """
    Get train/validation/test data for logit-propagated model.

    Args:
        data: `SemiSupervisedSingle` instance
        epsilon: value used in Le = I - (1 - epsilon)*normalized_adjacency.
        tol: value used in conjugate gradient to solve linear system.
        rescale: if True, rescale propagator by epsilon
        train_fun, validation_fun, test_fun: callables that map
            (propagator, node_features, labels, ids) to datasets / iterables.
    Returns:
        `SplitData` where each dataset is a single-example tuple of the form
            `((propagator, node_features), labels[ids])`
    """
    propagator = _get_propagator(data.graph, epsilon=epsilon, tol=tol, rescale=rescale)

    # jit so multiple input_propagated_data funs use the same propagator @ features
    @jax.jit
    def get_data():
        train_data, validation_data, test_data = (
            fun(propagator, data.node_features, data.labels, ids)
            for (fun, ids) in (
                (train_fun, data.train_ids),
                (validation_fun, data.validation_ids),
                (test_fun, data.test_ids),
            )
        )
        return SplitData(train_data, validation_data, test_data)

    return get_data()


# class InputPropagatedData:
#     def __init__(self, root):
#         self._root = root
#         self._graph = None

#     @staticmethod
#     def create(path: str, data: SemiSupervisedSingle):

#         assert not os.path.isfile(path)
#         with h5py.File(path, "w") as root:
#             root.create_dataset("labels", data.labels.to_py())
#             ids = root.create_group("ids")
#             for split in ("train", "validation", "test"):
#                 ids.create_dataset(split, data=getattr(data, f"{split}_ids").to_py())

#             # copy graph
#             graph = root.create_group("graph")
#             assert isinstance(data.graph, COO)
#             coo: COO = data.graph
#             graph.create_dataset("row", data=coo.row.to_py())
#             graph.create_dataset("col", data=coo.col.to_py())
#             graph.create_dataset("data", data=coo.data.to_py())

#             # input node features correspond to epsilon=1
#             root.create_group("node_features").create_dataset(
#                 "1.0", data=data.node_features.to_py()
#             )

#             root.attrs.update(
#                 {
#                     "num_nodes": data.num_nodes,
#                     "num_edges": data.num_edges,
#                     "num_classes": data.num_classes,
#                     "num_features": data.node_features.shape[1],
#                 }
#             )

#     @property
#     def graph(self) -> COO:
#         if self._graph is None:
#             graph = self._root["graph"]
#             self._graph = COO(
#                 (graph["data"][:], graph["row"][:], graph["col"][:]),
#                 shape=(self._root.attrs["num_nodes"],) * 2,
#             )
#         return self._graph

#     @property
#     def num_nodes(self) -> int:
#         return self._root.attrs["num_nodes"]

#     @property
#     def num_edges(self) -> int:
#         return self._root.attrs["num_edges"]

#     @property
#     def num_classes(self) -> int:
#         return self._root.attrs["num_classes"]

#     @property
#     def num_features(self) -> int:
#         return self._root.attrs["num_features"]

#     @property
#     def labels(self):
#         return self._root["labels"]

#     def ids(self, split: str):
#         return self._root["ids"][split]

#     @property
#     def train_ids(self):
#         return self.ids("train")

#     @property
#     def validation_ids(self):
#         return self.ids("validation")

#     @property
#     def test_ids(self):
#         return self.ids("test")

#     def require_propagated(self, epsilon: float, progbar: bool = True):
#         node_features = self._root["node_features"]
#         assert 0 < epsilon < 1, epsilon
#         key = str(float(epsilon))
#         if key not in node_features:
#             input_data = node_features["1.0"]
#             try:
#                 group = node_features.create_dataset(
#                     key, shape=input_data.shape, dtype=input_data.dtype
#                 )
#                 propagator = _get_propagator(self.graph, epsilon)
#                 cols = range(input_data.shape[1])
#                 desc = f"Propagating input features with epsilon={epsilon}"
#                 if progbar:
#                     cols = tqdm.tqdm(cols, desc=desc)
#                 else:
#                     print(desc)
#                 for col in cols:
#                     group[:, col] = (
#                         propagator @ jnp.asarray(input_data[:, col])
#                     ).to_py()
#                     if not progbar:
#                         print(f"Computed inverse {col+1} / {input_data.shape[1]}")

#             except (Exception, KeyboardInterrupt):
#                 del node_features[key]
#                 raise
#         return node_features[key]


# @configurable
# def create_input_propagated_data(
#     path: str, data_fn: tp.Callable[[], SemiSupervisedSingle]
# ):
#     if not os.path.isfile(path):
#         InputPropagatedData.create(path, data_fn())
#     return InputPropagatedData(h5py.File(path, "a"))


# @configurable
# class PropagatedDataset(huf.data.Dataset):
#     def __init__(
#         self,
#         data: InputPropagatedData,
#         batch_size: int,
#         split: str,
#         epsilon: tp.Union[float, tp.Iterable[float]],
#         shuffle_seed: tp.Optional[int] = None,
#         pad_to_batch_size: bool = True,
#     ):
#         if not hasattr(epsilon, "__iter__"):
#             epsilon = (epsilon,)
#         self.epsilon = tuple(epsilon)
#         self.data = data
#         self.batch_size = batch_size
#         self.split = split
#         self.shuffle_rng = (
#             None if shuffle_seed is None else np.random.default_rng(shuffle_seed)
#         )
#         self.ids = data.ids(split)
#         self.groups = tuple(self.data.require_propagated(eps) for eps in epsilon)
#         self.labels = self.data.labels
#         self.pad_to_batch_size = pad_to_batch_size
#         self.remainder = (
#             self.ids.size % self.batch_size if self.pad_to_batch_size else None
#         )

#     @property
#     def element_spec(self):
#         features = jax.core.ShapedArray(
#             shape=(self.batch_size, self.data.num_features * len(self.epsilon)),
#             dtype=jnp.float32,
#         )
#         labels = jax.core.ShapedArray(shape=(self.batch_size,), dtype=jnp.int32)
#         weights = jax.core.ShapedArray(shape=(self.batch_size,), dtype=jnp.int32)
#         return (features, labels, weights)

#     def __iter__(self):
#         def get_example(ids):
#             if self.shuffle_rng:
#                 ids = np.sort(ids)
#             features = jnp.concatenate([g[ids][:] for g in self.groups], axis=1)
#             labels = jnp.asarray(self.labels[ids])
#             yield (features, labels, weights[: ids.size])

#         ids = self.ids[:]
#         if self.shuffle_rng:
#             self.shuffle_rng.shuffle(ids)
#         weights = jnp.ones((self.batch_size,), dtype=jnp.float32)
#         for i in range(0, ids.size, self.batch_size):
#             yield get_example(ids[i : i + self.batch_size])

#         if self.remainder:
#             ids = ids[-self.remainder :]
#             features, labels, weights = get_example(ids)
#             pad = self.batch_size = self.remainder
#             features = jnp.pad(features, [[0, pad], [0, 0]])
#             labels = jnp.pad(labels, [[0, pad]])
#             weights = jnp.pad(weights[: ids.size], [[0, pad]])
#             yield (features, labels, weights)

#     def __len__(self):
#         n = len(self.ids)
#         return n // self.batch_size + int(self.remainder)

# def generator(
#     self,
#     epsilon: tp.Union[float, tp.Iterable[float]],
#     batch_size: int,
#     shuffle: bool,
#     split: str = "train",
# ) -> tp.Callable[[], jnp.ndarray]:
#     if isinstance(epsilon, float):
#         epsilon = (epsilon,)
#     node_features = [self.require_propagated(eps) for eps in epsilon]


def create_cached_labels(
    path: str, data: SemiSupervisedSingle,
):
    print("Creating cached labels...")
    try:
        with h5py.File(path, "w") as root:
            root.create_dataset("train", data=data.labels[data.train_ids].to_py())
            root.create_dataset(
                "validation", data=data.labels[data.validation_ids].to_py()
            )
            root.create_dataset("test", data=data.labels[data.test_ids].to_py())
    except (StopIteration, KeyboardInterrupt):
        if os.path.isfile(path):
            os.remove(path)
        raise


def create_cached_features(
    path: str,
    data: SemiSupervisedSingle,
    epsilon: float,
    tol: float = 1e-5,
    maxiter: tp.Optional[int] = None,
    progbar: bool = True,
):
    print("Creating cached features...")
    print(f"epsilon = {epsilon}")
    print(f"tol     = {tol}")
    print(f"maxiter = {maxiter}")
    try:
        with h5py.File(path, "w") as root:
            root.attrs["epsilon"] = epsilon
            root.attrs["tol"] = tol
            root.attrs["maxiter"] = maxiter
            propagator = _get_propagator(
                data.graph, epsilon=epsilon, tol=tol, maxiter=maxiter
            )
            features = data.node_features
            num_features = features.shape[1]

            splits = ("train", "validation", "test")
            ids = tuple(getattr(data, f"{split}_ids").to_py() for split in splits)
            datasets = tuple(
                root.create_dataset(
                    split, shape=(ids_.size, num_features), dtype=features.dtype
                )
                for (split, ids_) in zip(splits, ids)
            )

            indices = range(features.shape[1])
            if progbar:
                indices = tqdm.tqdm(indices, "Propagating features...")
            for i in indices:
                result = (propagator @ jnp.asarray(features[:, i])).to_py()
                for dataset, ids_ in zip(datasets, ids):
                    dataset[:, i] = result[ids_]
    except (Exception, KeyboardInterrupt):
        if os.path.isfile(path):
            os.remove(path)
        raise


def _load_cached(path: str):
    path = os.path.expanduser(os.path.expandvars(path))
    assert path.endswith(".h5"), path
    assert os.path.isfile(path)
    return h5py.File(path, "r")


class BatchedDataset(huf.data.Dataset):
    def __init__(
        self, features: tp.Iterable[h5py.Group], labels: h5py.Group, batch_size: int
    ):
        self._features = tuple(features)
        assert len(self._features) > 0, len(self._features)
        self._labels = labels
        self._batch_size = batch_size

        self._num_examples = self._labels.shape[0]
        for f in self._features:
            assert f.shape[0] == self._num_examples, (f.shape[0], self._num_examples)
        self._size = self._num_examples // batch_size
        self._feature_size = sum(f.shape[1] for f in self._features)

    def _get(self, p):
        features = tuple(jnp.asarray(f[p]) for f in self._features)
        features = jnp.concatenate(features, axis=1)
        labels = jnp.asarray(self._labels[p])
        return features, labels

    def __len__(self) -> int:
        return self._size

    @property
    def element_spec(self):
        features = jax.core.ShapedArray(
            (self._batch_size, self._feature_size), jnp.float32
        )
        labels = jax.core.ShapedArray((self._batch_size,), jnp.int32)
        return features, labels

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError("Abstract method")


class TrainDataset(BatchedDataset):
    def __init__(
        self,
        features: tp.Iterable[h5py.Group],
        labels: h5py.Group,
        batch_size: int,
        shuffle_rng=0,
    ):
        def get_features(group):
            if hasattr(group, "train"):
                return group["train"]
            return group

        super().__init__((get_features(g) for g in features), labels, batch_size)
        self._rng = PRNGSequence(shuffle_rng)

    def __len__(self) -> int:
        return self._size

    def __iter__(self):
        perm = jax.random.permutation(next(self._rng), self._num_examples)

        def gen():
            for start in range(0, self._size * self._batch_size, self._batch_size):
                p = jnp.sort(perm[start : start + self._batch_size]).to_py()
                yield self._get(p)

        return iter(gen())


class TestDataset(BatchedDataset):
    def __init__(
        self, features: tp.Iterable[h5py.Group], labels: h5py.Group, batch_size: int,
    ):
        super().__init__(features, labels, batch_size)
        # manage possible final batch
        self._final_size = self._num_examples % self._batch_size
        if self._final_size:
            self._size += 1

    @property
    def element_spec(self):
        # add sample weight
        features, labels = super().element_spec
        sample_weights = jax.core.ShapedArray((self._batch_size,), jnp.float32)
        return features, labels, sample_weights

    def __iter__(self):
        sample_weight = jnp.ones((self._batch_size,), jnp.float32)

        def gen():
            for start in range(
                0, self._num_examples - self._final_size, self._batch_size
            ):
                features, labels = self._get(slice(start, start + self._batch_size))
                yield features, labels, sample_weight
            if self._final_size:
                features, labels = self._get(slice(-self._final_size, None))
                padding = [0, self._batch_size - self._final_size]
                features = jnp.pad(features, [padding, [0, 0]])
                labels = jnp.pad(labels, [padding])
                weights = jnp.pad(sample_weight[: self._final_size], [padding])
                yield (features, labels, weights)

        return iter(gen())


def _get_batched_split_data(
    features: tp.Sequence[h5py.Group], labels: h5py.Group, batch_size: int
):
    train_ds = TrainDataset((f["train"] for f in features), labels["train"], batch_size)
    val_ds = TestDataset(
        (f["validation"] for f in features), labels["validation"], batch_size
    )
    test_ds = TestDataset((f["test"] for f in features), labels["test"], batch_size)
    return SplitData(train_ds, val_ds, test_ds)


def _features_path(root_dir: str, epsilon: float, maxiter: int, tol: float):
    if epsilon == 1:
        return os.path.join(root_dir, "base_features.h5")
    return os.path.join(
        root_dir, f"eps={epsilon:.1e}_tol={tol:.1e}_maxiter={maxiter}.h5"
    )


def _labels_path(root_dir: str):
    return os.path.join(root_dir, "labels.h5")


@configurable
def create_cached_input_propagated_data(
    root_dir: str,
    data_fn: tp.Callable[[], SemiSupervisedSingle],
    epsilon: tp.Union[tp.Iterable[float], float],
    maxiter: tp.Optional[int] = None,
    tol: float = 1e-5,
    progbar: bool = True,
):
    os.makedirs(root_dir, exist_ok=True)
    if not hasattr(epsilon, "__iter__"):
        epsilon = (epsilon,)

    root_dir = os.path.expanduser(os.path.expandvars(root_dir))
    data = None
    labels_path = _labels_path(root_dir)
    if not os.path.isfile(labels_path):
        data = data_fn()
        create_cached_labels(labels_path, data)

    for eps in epsilon:
        features_path = _features_path(root_dir, eps, maxiter, tol)
        if not os.path.isfile(features_path):
            if data is None:
                data = data_fn()
            create_cached_features(
                features_path, data, eps, tol=tol, maxiter=maxiter, progbar=progbar
            )


@configurable
def default_cache_dir(base_dir: tp.Optional[str] = None, data_name: str = gin.REQUIRED):
    if base_dir is None:
        base_dir = os.environ.get("IGCN_DATA", "~/igcn-data")
    root_dir = os.path.join(base_dir, data_name)
    return os.path.expanduser(os.path.expandvars(root_dir))


@configurable
def get_cached_input_propagated_data(
    root_dir: str,
    data_fn: tp.Callable[[], SemiSupervisedSingle],
    epsilon: tp.Union[float, tp.Iterable[float]],
    batch_size: tp.Optional[int],
    tol: float = 1e-2,
    maxiter: tp.Optional[int] = 1000,
    progbar: bool = True,
    allow_creation: bool = False,
) -> SplitData:
    root_dir = os.path.expanduser(os.path.expandvars(root_dir))
    if not hasattr(epsilon, "__iter__"):
        epsilon = (epsilon,)
    if allow_creation:
        create_cached_input_propagated_data(
            root_dir,
            data_fn,
            epsilon=epsilon,
            tol=tol,
            maxiter=maxiter,
            progbar=progbar,
        )

    def load_features(epsilon):
        path = _features_path(root_dir, epsilon, maxiter, tol)
        assert os.path.isfile(path), path
        features = _load_cached(path)

        def assert_consistent(key, value):
            assert features.attrs[key] == value, (key, features.attrs[key], value)

        if epsilon != 1:
            assert_consistent("epsilon", epsilon)
            assert_consistent("tol", tol)
            assert_consistent("maxiter", maxiter)
        return features

    features = tuple(load_features(eps) for eps in epsilon)
    path = _labels_path(root_dir)
    assert os.path.isfile(path), path
    labels = _load_cached(path)
    return _get_batched_split_data(features, labels, batch_size)
