import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config
from spax import COO, ops
from spax.utils import diag as diags

from grax import test_utils

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


class TestUtilsTest(jtu.JaxTestCase):
    def test_random_adjacency(self):
        num_nodes = 10
        num_edges = 20
        dtype = jnp.float32
        adj = test_utils.random_adjacency(
            jax.random.PRNGKey(0), num_nodes, num_edges, dtype=dtype
        )
        self._test_adjacency(adj)

    def test_star_adjacency(self):
        num_nodes = 20
        dtype = jnp.float32
        adj = test_utils.star_adjacency(num_nodes, dtype=dtype)
        self._test_adjacency(adj)

    def _test_adjacency(self, adj: COO):
        self.assertAllClose(adj.data, jnp.ones_like(adj.data))
        adj = adj.todense()
        # ensure symmetric
        self.assertAllClose(adj, adj.T)
        # ensure zeros on diagonal
        self.assertAllClose(jnp.diag(adj), jnp.zeros((adj.shape[0],), dtype=adj.dtype))
        adj = ops.subtract(diags(ops.sum(adj, axis=1)), adj)
        w, v = jnp.linalg.eigh(adj)
        self.assertAllClose(w[0], jnp.zeros((), dtype=w.dtype))
        self.assertTrue(jnp.all(w[1] > 0))  # connected
        # uniform zero eigenvector
        self.assertAllClose(jnp.tile(v[:1, 0], (v.shape[0] - 1,)), v[1:, 0])


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
