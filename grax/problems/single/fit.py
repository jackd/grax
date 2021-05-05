import time
import typing as tp
from collections import defaultdict
from functools import partial

import gin
import numpy as np

import haiku as hk
import huf
import jax
import jax.numpy as jnp
import spax
from grax.problems.single import data
from huf.experiments import ExperimentCallback, experiment_context
from huf.metrics import get_combined_metrics
from huf.types import FitState, Metrics, Splits
from jax.experimental.sparse_ops import COO

configurable = partial(gin.configurable, module="grax.problems.single")


@configurable
def semi_supervised_net_fun(
    inputs, is_training: bool, *, module_fun: tp.Callable[[], hk.Module], **kwargs
):
    # graph, node_features = inputs
    data, row, col, node_features = inputs
    n = node_features.shape[0]
    graph = COO((data, row, col), shape=(n, n))
    module = module_fun(**kwargs)
    return module(graph, node_features, is_training)


@configurable
def as_example(
    data: data.SemiSupervisedSingle, split: str = Splits.TRAIN, dtype=jnp.float32
):
    # not sure why we need to unpack/repack graph, but avoids errors in GAT
    if split == Splits.TRAIN:
        mask = data.train_mask
    elif split == Splits.VALIDATION:
        mask = data.validation_mask
    elif split == Splits.TEST:
        mask = data.test_mask
    else:
        raise ValueError(f"Invalid split {split}.")
    graph = data.graph
    graph = spax.ops.to_coo(graph)

    def cast(x):
        return x.astype(dtype)

    # node_features = jax.tree_util.tree_map(cast, data.node_features)
    # return graph, node_features, cast(mask)

    inputs = (
        cast(graph.data),
        graph.row,
        graph.col,
        jax.tree_util.tree_map(cast, data.node_features),
    )
    return (inputs, data.labels, cast(mask))


def _fit_semi_supervised(
    model: huf.models.Model,
    initial_state: tp.Union[int, huf.types.PRNGKey, FitState],
    train_example,
    validation_example,
    test_example,
    steps,
    callbacks,
    verbose: bool,
) -> Metrics:
    t = time.time()
    result = model.fit(
        initial_state,
        (train_example,),
        epochs=steps,
        validation_data=(validation_example,),
        callbacks=callbacks,
        verbose=verbose,
    )
    dt = time.time() - t
    print(f"Training took {dt:.4f}s")
    model_state = result.state.model_state
    test_metrics = model.evaluate(model_state, [test_example])
    metrics = get_combined_metrics(
        result.train_metrics, result.validation_metrics, test_metrics
    )
    return metrics


@configurable
def fit_semi_supervised(
    model: huf.models.Model,
    initial_state: tp.Union[int, huf.types.PRNGKey, FitState],
    data: data.SemiSupervisedSingle,
    steps: int,
    callbacks: tp.Iterable[huf.callbacks.Callback] = (),
    verbose: bool = True,
    dtype: jnp.dtype = jnp.float32,
) -> Metrics:
    train_example = as_example(data, Splits.TRAIN, dtype=dtype)
    validation_example = as_example(data, Splits.VALIDATION, dtype=dtype)
    test_example = as_example(data, Splits.TEST, dtype=dtype)

    return _fit_semi_supervised(
        model,
        initial_state,
        train_example,
        validation_example,
        test_example,
        steps,
        callbacks,
        verbose,
    )


@configurable
def print_moments(
    results: tp.Iterable[Metrics], print_fun: tp.Callable[[str], None] = print
):
    out = defaultdict(list)
    for r in results:
        for k, v in r.items():
            out[k].append(v)
    max_len = max(len(k) for k in out)
    for k in sorted(out):
        v = out[k]
        print_fun(f"{k.ljust(max_len)} = {np.mean(v):.5f} +- {np.std(v):.5f}")


@configurable
def fit_many_semi_supervised(
    num_repeats: int,
    model: huf.models.Model,
    rng: tp.Union[int, huf.types.PRNGKey],
    data: data.SemiSupervisedSingle,
    steps: int,
    callbacks: tp.Iterable[huf.callbacks.Callback] = (),
    verbose: bool = True,
    dtype: jnp.dtype = jnp.float32,
    experiment_callbacks: tp.Iterable[ExperimentCallback] = (),
) -> tp.List[Metrics]:
    assert isinstance(rng, (int, huf.types.PRNGKey))
    if isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)

    train_example = as_example(data, Splits.TRAIN, dtype=dtype)
    validation_example = as_example(data, Splits.VALIDATION, dtype=dtype)
    test_example = as_example(data, Splits.TEST, dtype=dtype)

    results = []
    for i, s in enumerate(jax.random.split(rng, num_repeats)):
        with experiment_context(experiment_callbacks):
            result = _fit_semi_supervised(
                model,
                s,
                train_example,
                validation_example,
                test_example,
                steps,
                callbacks,
                verbose=verbose and i == 0,
            )
        for callback in experiment_callbacks:
            callback.on_done(result)
        results.append(result)

    return results
