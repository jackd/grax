import typing as tp

import jax
import jax.numpy as jnp
from huf.types import PRNGKey


def sample_per_class(
    rng: PRNGKey,
    labels: jnp.ndarray,
    examples_per_class: int,
    valid_indices: tp.Optional[jnp.ndarray] = None,
    num_classes: tp.Optional[int] = None,
) -> jnp.ndarray:
    assert labels.ndim == 1, labels.shape
    if valid_indices is None:
        valid_indices = jnp.arange(labels.size, dtype=jnp.int64)
        valid_labels = labels
    else:
        assert valid_indices.ndim == 1, valid_indices.shape
        valid_labels = labels[valid_indices]

    if num_classes is None:
        num_classes = jnp.max(labels) + 1
    all_class_indices = []
    for class_index, key in enumerate(jax.random.split(rng, num_classes)):
        (class_indices,) = jnp.where(valid_labels == class_index)
        assert class_indices.size >= examples_per_class
        class_indices = jax.random.choice(
            key, class_indices, (examples_per_class,), replace=False
        )
        class_indices = valid_indices[class_indices]
        all_class_indices.append(class_indices)
    class_indices = jnp.concatenate(all_class_indices)
    class_indices = jnp.sort(class_indices)
    return class_indices


class IndexSplits(tp.NamedTuple):
    splits: tp.Mapping[str, jnp.ndarray]
    remaining: jnp.ndarray


def get_index_splits(
    rng: PRNGKey,
    labels: jnp.ndarray,
    examples_per_class: tp.Mapping[str, int],
    valid_indices: tp.Optional[jnp.ndarray] = None,
    num_classes: tp.Optional[int] = None,
) -> IndexSplits:
    keys = jax.random.split(rng, len(examples_per_class))
    if valid_indices is None:
        valid_indices = jnp.arange(labels.shape[0])

    if num_classes is None:
        num_classes = jnp.max(labels) + 1
    out = {}
    for key, k in zip(keys, sorted(examples_per_class)):
        sample_indices = sample_per_class(
            key, labels, examples_per_class[k], valid_indices, num_classes
        )
        valid_indices = jnp.setdiff1d(valid_indices, sample_indices, assume_unique=True)
        out[k] = sample_indices
    return IndexSplits(out, valid_indices)


def split_by_class(
    rng: PRNGKey,
    labels: jnp.ndarray,
    examples_per_class: tp.Sequence[int],
    num_classes: tp.Optional[int] = None,
) -> tp.Sequence[jnp.ndarray]:
    splits = jnp.cumsum(jnp.asarray(examples_per_class))
    del examples_per_class
    if num_classes is None:
        num_classes = labels.max() + 1
    masks = jax.nn.one_hot(labels, num_classes, dtype=bool)
    id_lists = [[] for _ in range(len(splits) + 1)]
    for i, class_rng in enumerate(jax.random.split(rng, num_classes)):
        (indices,) = jnp.where(masks[:, i])
        indices = jax.random.permutation(class_rng, indices)
        indices = jnp.split(indices, jnp.minimum(splits, indices.size))
        for id_list, ids in zip(id_lists, indices):
            id_list.append(ids)
    return tuple(jnp.sort(jnp.concatenate(ids)) for ids in id_lists)
