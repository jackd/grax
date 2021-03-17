import jax
import jax.numpy as jnp
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from grax.jax_utils import linalg

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


class LinalgTest(jtu.JaxTestCase):
    def test_lu(self):
        m = 10
        n = 100
        seed = 0
        dtype = jnp.float64
        x = jax.random.uniform(jax.random.PRNGKey(seed), (m, n), dtype=dtype)
        l, q = linalg.lq(x)
        self.assertAllClose(l @ q, x, atol=1e-4)
        jtu.check_grads(linalg.lq, (x,), modes=["rev"], order=1)

    def test_qr(self):
        m = 100
        n = 10
        seed = 0
        dtype = jnp.float64
        x = jax.random.uniform(jax.random.PRNGKey(seed), (m, n), dtype=dtype)
        q, r = linalg.qr(x)
        self.assertAllClose(q @ r, x, atol=1e-4)
        jtu.check_grads(linalg.qr, (x,), modes=["rev"], order=1)


if __name__ == "__main__":
    # LinalgTest().test_lu()
    absltest.main(testLoader=jtu.JaxTestLoader())
