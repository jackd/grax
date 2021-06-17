import functools
import typing as tp

import gin
import jax
import jax.numpy as jnp
from jax.experimental.sparse_ops import COO

import haiku as hk
import spax
from grax.graph_utils.transforms import symmetric_normalize
from grax.hk_utils import mlp
from spax.linalg import linear_operators as lin_ops

configurable = functools.partial(gin.configurable, module="igat")


def _get_shifted_laplacian(adj: COO, epsilon: float):
    x = adj
    x = spax.ops.scale(symmetric_normalize(x), -(1 - epsilon))
    # x64 required for medium-sized examples, e.g. ogbn-arxiv
    # with jax.experimental.enable_x64():
    x = spax.ops.add(spax.eye(x.shape[0], x.dtype, x.row.dtype), x)
    return x


def _get_propagator(
    adj: COO, epsilon: float, tol: float = 1e-5
) -> lin_ops.LinearOperator:
    x = _get_shifted_laplacian(adj, epsilon)
    return lin_ops.SelfAdjointInverse(
        lin_ops.MatrixWrapper(x, is_self_adjoint=True), tol=tol
    )


class IGAT(hk.Module):
    def __init__(
        self,
        out_size: int,
        attention_size: int = 64,
        tol: float = 1e-5,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.attention_size = attention_size
        self.out_size = out_size
        self.tol = tol

    def __call__(self, row, col, features):
        attn = hk.Linear(self.attention_size)(features)
        vals = jax.nn.sigmoid(jnp.einsum("na,na->n", attn[row], attn[col]))
        adj = COO((vals, row, col), shape=(features.shape[0],) * 2)
        epsilon = jax.nn.sigmoid(
            hk.get_parameter(
                "epsilon_sig_inv",
                shape=(),
                dtype=features.dtype,
                init=lambda _, dtype: jnp.asarray(-2.0, dtype=dtype),
            )
        )
        propagator = _get_propagator(adj, epsilon, self.tol)
        features = hk.Linear(self.out_size)(features)
        return propagator @ features


class MultiHeadedIGAT(hk.Module):
    def __init__(
        self,
        num_heads: int,
        out_size: int,
        attention_size: int = 64,
        tol: float = 1e-5,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.out_size = out_size
        self.attention_size = attention_size
        self.tol = tol

    def __call__(self, row, col, features):
        attn = hk.Linear(self.num_heads * self.attention_size)(features)
        attn = jnp.reshape(attn, (-1, self.num_heads, self.attention_size))
        attn = jax.nn.sigmoid(jnp.einsum("eha,eha->eh", attn[row], attn[col]))
        epsilon = jax.nn.sigmoid(
            hk.get_parameter(
                "epsilon_sig_inv",
                shape=(self.num_heads,),
                dtype=features.dtype,
                init=lambda shape, dtype: jnp.full(shape, -2.0, dtype=dtype),
            )
        )
        features = hk.Linear(self.num_heads * self.out_size)(features)
        features = jnp.reshape(features, (-1, self.num_heads, self.out_size))

        def propagate(features, attn, eps, row, col):
            adj = COO((attn, row, col), shape=(features.shape[0],) * 2)
            propagator = _get_propagator(adj, eps, self.tol)
            return propagator @ features
            # shifted_lap = _get_shifted_laplacian(adj, eps)
            # return jax.scipy.sparse.linalg.cg(lambda x: shifted_lap @ x, features)

        # out_ax = 0
        # out = jax.vmap(propagate, in_axes=(1, 1, 0, None, None), out_axes=out_ax)(
        #     features, attn, epsilon, row, col
        # )
        def unstack(x, axis):
            return [x.take(i, axis=axis) for i in range(x.shape[axis])]

        features = unstack(features, axis=1)
        attn = unstack(attn, axis=1)
        eps = unstack(epsilon, axis=0)
        out = [propagate(f, a, e, row, col) for (f, a, e) in zip(features, attn, eps)]
        return sum(out) / len(out)
        # return out.sum(axis=out_ax)


@configurable
class IGATNet(hk.Module):
    def __init__(
        self,
        num_classes: int,
        num_heads: int = 1,
        *,
        node_transform: tp.Callable = mlp,
        name: tp.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name)
        self.node_transform = node_transform
        if num_heads == 1:
            self.igat = IGAT(out_size=num_classes, **kwargs)
        else:
            self.igat = MultiHeadedIGAT(
                out_size=num_classes, num_heads=num_heads, **kwargs
            )

    def __call__(self, row, col, features, is_training: bool):
        features = self.node_transform(features, is_training=is_training)
        return self.igat(row, col, features)
