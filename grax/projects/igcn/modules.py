import typing as tp
from functools import partial

import gin
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import spax
from jax.experimental.sparse import COO
from spax.linalg import linear_operators as lin_ops

from grax.hk_utils import mlp
from grax.projects.igcn.data import ids_to_mask_matrix

configurable = partial(gin.configurable, module="igcn")


def hstack(a, b):
    if all(isinstance(x, jnp.ndarray) for x in (a, b)):
        return jnp.concatenate((a, b), axis=1)
    return lin_ops.HStacked(a, b)


@configurable
class IGCN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        head_transform: tp.Optional[
            tp.Callable[[jnp.ndarray, bool], jnp.ndarray]
        ] = mlp,
        tail_transform: tp.Optional[
            tp.Callable[[jnp.ndarray, bool], jnp.ndarray]
        ] = None,
        smooth_only: bool = True,
        name=None,
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.smooth_only = smooth_only
        self.head_transform = head_transform
        self.tail_transform = tail_transform

    def __call__(
        self,
        smoother: tp.Any,  # anything with a __matmul__ operator
        node_features: jnp.ndarray,
        ids: jnp.ndarray,
        is_training: bool = False,
    ):
        if self.head_transform is None:
            x = node_features
        else:
            x = self.head_transform(node_features, is_training=is_training, ids=ids)
        if self.tail_transform is None:
            stddev = 1.0 / np.sqrt(x.shape[-1] * (1 if self.smooth_only else 2))
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            if x.shape[1] > self.num_classes:
                x = hk.Linear(self.num_classes, w_init=w_init)(x)
                logits = smoother @ x
            else:
                logits = hk.Linear(self.num_classes, w_init=w_init)(smoother @ x)
            if not self.smooth_only:
                logits += hk.Linear(self.num_classes, w_init=w_init, with_bias=False)(
                    x[ids]
                )
        else:
            tail_x = smoother @ x
            if not self.smooth_only:
                tail_x = hstack(tail_x, x[ids])
            tail_x = self.tail_transform(tail_x, is_training=is_training)
            logits = hk.Linear(self.num_classes)(tail_x)
        return logits


@jax.tree_util.register_pytree_node_class
class ModifiedLaplacian(lin_ops.LinearOperator):
    def __init__(self, adj: COO, epsilon: jnp.ndarray):
        self._adj = adj
        self._epsilon = epsilon

    def tree_flatten(self):
        return (self._adj, self._epsilon), {}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def _matmul(self, x):
        return x - (1 - self._epsilon) * (self._adj @ x)

    @property
    def dtype(self) -> jnp.dtype:
        return self._adj.dtype

    @property
    def shape(self) -> tp.Tuple[int, ...]:
        return self._adj.shape

    @property
    def is_self_adjoint(self) -> bool:
        return True


@configurable
def if_e0(k0: str, k1: str, values: jnp.ndarray):
    """Useful as a `predicate` in `optax_utils.partition`."""
    del k0, values
    return k1 == "e0"


@configurable
class LearnedIGCN(IGCN):
    def __init__(
        self,
        *args,
        tol: tp.Union[float, tp.Union[float, float]] = 1e-2,
        maxiter: tp.Optional[int] = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._tol = tol
        self._maxiter = maxiter

    def __call__(
        self,
        adj: COO,
        node_features: jnp.ndarray,
        ids: jnp.ndarray,
        is_training: bool = False,
    ):
        e0 = hk.get_parameter("e0", (), adj.dtype, jnp.zeros)
        epsilon = jax.nn.sigmoid(e0)
        tol = (
            self._tol[int(not is_training)]
            if hasattr(self._tol, "__iter__")
            else self._tol
        )
        maxiter = (
            self._maxiter[int(not is_training)]
            if hasattr(self._maxiter, "__iter__")
            else self._maxiter
        )

        def smooth(x):
            def f(x):
                return x - (1 - epsilon) * (adj @ x)

            assert x.ndim == 2
            if x.shape[1] > ids.size:
                # (M.T Le^{-1}) X
                M = ids_to_mask_matrix(ids, adj.shape[0], dtype=adj.dtype)
                MLT = jax.scipy.sparse.linalg.cg(f, M, tol=tol, maxiter=maxiter)[0]
                return MLT.T @ spax.ops.to_dense(x)

            # M.T @ (Le^{-1} X)
            return jax.scipy.sparse.linalg.cg(
                f, spax.ops.to_dense(x), tol=tol, maxiter=maxiter
            )[0][ids]

        if self.head_transform is None:
            x = node_features
        else:
            x = self.head_transform(node_features, is_training=is_training, ids=ids)
        if self.tail_transform is None:
            stddev = 1.0 / np.sqrt(x.shape[-1] * (1 if self.smooth_only else 2))
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            if x.shape[1] > self.num_classes:
                x = hk.Linear(self.num_classes, w_init=w_init)(x)
                logits = smooth(x)
            else:
                logits = hk.Linear(self.num_classes, w_init=w_init)(smooth(x))
            if not self.smooth_only:
                logits += hk.Linear(self.num_classes, w_init=w_init, with_bias=False)(
                    x[ids]
                )
        else:
            tail_x = smooth(x)
            if not self.smooth_only:
                tail_x = hstack(tail_x, spax.ops.to_dense(x)[ids])
            tail_x = self.tail_transform(tail_x, is_training=is_training)
            logits = hk.Linear(self.num_classes)(tail_x)
        return logits
