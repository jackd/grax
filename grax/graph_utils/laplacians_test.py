import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from grax.data import laplacians as lap
from grax.data import test_utils
from spax.linalg.utils import standardize_signs

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


class LaplaciansTest(jtu.JaxTestCase):
    def test_laplacian(self):
        num_nodes = 10
        num_edges = 20
        dtype = jnp.float32
        adj = test_utils.random_adjacency(
            jax.random.PRNGKey(0), num_nodes, num_edges, dtype=dtype
        )
        L, row_sum = lap.laplacian(adj)
        L = L.todense()
        self.assertAllClose(jnp.diag(L), row_sum)
        self.assertAllClose(L, L.T)
        w, v = jnp.linalg.eigh(L)
        v = standardize_signs(v)
        self.assertAllClose(w[0], jnp.asarray(0.0, dtype))
        self.assertTrue(w[1] > 0)  # ensure connected
        # correct zero-eigenvector
        self.assertAllClose(
            v[:, 0], jnp.ones((num_nodes,), dtype) / jnp.sqrt(num_nodes).astype(dtype),
        )

    def test_normalized_laplacian(self):
        num_nodes = 10
        num_edges = 20
        dtype = jnp.float32
        adj = test_utils.random_adjacency(
            jax.random.PRNGKey(0), num_nodes, num_edges, dtype=dtype
        )
        L, row_sum = lap.normalized_laplacian(adj)
        L = L.todense()
        self.assertAllClose(jnp.diag(L), jnp.ones((num_nodes,), dtype=dtype))
        self.assertAllClose(L, L.T)
        w, v = jnp.linalg.eigh(L)
        eps = 1e-5
        # ensure eigenvalues are all in [0, 2]
        self.assertTrue(jnp.all(w > -eps))
        self.assertTrue(jnp.all(w <= 2 + eps))
        self.assertAllClose(w[0], jnp.asarray(0.0, dtype), rtol=1e-10)
        # correct zero-eigenvector
        self.assertAllClose(v[:, 0], lap.normalized_laplacian_zero_eigenvector(row_sum))
        self.assertTrue(w[1] > eps)  # ensure connected

    def test_shifted_normalized_laplacian(self):
        num_nodes = 10
        num_edges = 20
        dtype = jnp.float32
        adj = test_utils.random_adjacency(
            jax.random.PRNGKey(0), num_nodes, num_edges, dtype=dtype
        )
        L, row_sum = lap.normalized_laplacian(adj, shift=2.0)
        L = L.todense()
        self.assertAllClose(jnp.diag(L), -jnp.ones((num_nodes,), dtype=dtype))
        self.assertAllClose(L, L.T)
        w, v = jnp.linalg.eigh(L)
        eps = 1e-5
        # ensure eigenvalues are all in [-2, 0]
        self.assertTrue(jnp.all(w > -2 - eps))
        self.assertTrue(jnp.all(w <= eps))
        self.assertAllClose(w[0], jnp.asarray(-2.0, dtype), rtol=1e-10)
        self.assertAllClose(v[:, 0], lap.normalized_laplacian_zero_eigenvector(row_sum))


if __name__ == "__main__":
    # LaplaciansTest().test_random_edges()
    # LaplaciansTest().test_laplacian()
    # LaplaciansTest().test_normalized_laplacian()
    # LaplaciansTest().test_deflate_constant()
    # LaplaciansTest().test_shifted_normalized_laplacian()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
