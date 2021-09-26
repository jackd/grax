import typing as tp
from functools import partial

import gin
import haiku as hk
import jax
import jax.numpy as jnp
from deqx.deq import DEQ, SolverType
from deqx.fpi import fpi_with_vjp
from huf.module_ops import Linear, dropout
from jax.experimental.sparse.ops import COO

from grax.projects.gcn.modules import GraphConvolution
from grax.types import Activation

configurable = partial(gin.configurable, module="deq_gcn")


def layer_norm(x):
    return hk.LayerNorm(-1, True, True)(x)


@configurable
def igcn_fn(h, x, graph_v, graph_r, graph_c, epsilon=0.1, use_layer_norm: bool = True):
    graph = COO((graph_v, graph_r, graph_c), shape=(x.shape[0],) * 2)
    out = (1 - epsilon) * (graph @ h) + x
    if use_layer_norm:
        out = layer_norm(out)
    return out


@configurable
def graph_resnet_simple(
    w_init_std: float = 1e-2,
    activation: tp.Callable = jax.nn.relu,
    use_layer_norm: bool = True,
):
    w_init = hk.initializers.TruncatedNormal(w_init_std)

    def f(z, x, *graph_components):
        A = COO(graph_components, shape=(x.shape[0],) * 2)
        z = GraphConvolution(x.shape[-1], kernel_initializer=w_init)(A, z)
        z = activation(z + x)
        if use_layer_norm:
            return layer_norm(z)
        return z

    return f


@configurable
class DEQGCN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_filters: int = 64,
        dropout_rate: float = 0.5,
        deq_fn: tp.Callable = graph_resnet_simple(),
        final_activation: Activation = lambda x: x,
        maxiter: int = 256,
        use_layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.hidden_filters = hidden_filters
        self.final_activation = final_activation
        self.maxiter = maxiter
        self.deq_fn = deq_fn
        self.use_layer_norm = use_layer_norm

    def __call__(
        self,
        graph: tp.Union[jnp.ndarray, COO],
        node_features: jnp.ndarray,
        is_training: tp.Optional[bool] = None,
    ):
        node_features = dropout(node_features, self.dropout_rate, is_training)
        node_features = Linear(self.hidden_filters)(node_features)
        node_features = jax.nn.relu(node_features)
        if self.use_layer_norm:
            node_features = layer_norm(node_features)
        node_features = dropout(node_features, self.dropout_rate, is_training)

        h = jnp.zeros_like(node_features)
        graph_components = graph.data, graph.row, graph.col
        node_features = DEQ(
            self.deq_fn,
            partial(
                fpi_with_vjp,
                tol=1e-5,
                maxiter=self.maxiter,
                jacobian_solver=partial(
                    jax.scipy.sparse.linalg.gmres, tol=1e-5, maxiter=self.maxiter
                ),
            ),
            solver_type=SolverType.FPI,
        )(h, node_features, *graph_components)
        node_features = dropout(node_features, self.dropout_rate, is_training)
        preds = hk.Linear(self.num_classes)(node_features)
        preds = self.final_activation(preds)
        return preds
