import operator
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental.sparse_ops import JAXSparse


@jax.tree_util.register_pytree_node_class
class MatrixInverse:
    def __init__(self, A: JAXSparse, **kwargs):
        assert isinstance(A, JAXSparse)
        assert A.shape[0] == A.shape[1]
        self.A = A
        self._cg_kwargs = kwargs

    @property
    def shape(self) -> tp.Tuple[int, int]:
        return self.A.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.A.dtype

    @property
    def ndim(self) -> int:
        return 2

    def _apply(self, x):
        assert x.shape[0] == self.shape[1], (x.shape, self.shape)
        return jax.scipy.sparse.linalg.cg(
            partial(operator.matmul, self.A), x, **self._cg_kwargs
        )[0]

    def __matmul__(self, x):
        return self._apply(x)

    def matmul(self, x):
        return self._apply(x)

    def matvec(self, x):
        return self._apply(x)

    def todense(self):
        return self._apply(jnp.eye(self.shape[0], dtype=self.dtype))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        assert len(children) == 1
        return MatrixInverse(children[0], **aux_data)

    def tree_flatten(self):
        return (self.A,), self._cg_kwargs
