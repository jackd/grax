import os
import tempfile

import jax
import jax.profiler
from huf.ops import sparse_categorical_crossentropy

from grax.problems.single.data import get_data
from grax.projects.igcn.data import get_propagator

problem = "cora"
epsilon = 0.1
seed = 0

data = get_data(problem)
num_classes = data.labels.max() + 1
num_nodes = data.graph.shape[0]
x = jax.random.normal(jax.random.PRNGKey(seed), shape=(num_nodes, num_classes))
prop = get_propagator(data.graph, epsilon=epsilon)


def fn(x, prop, labels):
    logits = prop @ x
    loss = sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return loss.sum()


@jax.jit
def outputs_fn(x, prop, labels):
    grad = jax.grad(fn, 0)(x, prop, labels)
    return grad


for _ in range(10):
    outputs = outputs_fn(x, prop, data.labels)
# outputs = (prop, x)

jax.tree_util.tree_map(lambda x: x.block_until_ready(), outputs)

folder = os.path.join(tempfile.gettempdir(), "grax-igcn-benchmarks")
os.makedirs(folder, exist_ok=True)
path = os.path.join(folder, "memory.prof")

jax.profiler.save_device_memory_profile(path)
print(f"Memory profile saved to {path}. Open with")
print(f"pprof --web {path}")
