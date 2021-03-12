import typing as tp
from functools import partial

import gin
import jax
import jax.numpy as jnp

import haiku as hk
import huf
import spax
from grax.problems.single import data

configurable = partial(gin.configurable, module="grax.problems.single")


@configurable
def semi_supervised_net_fun(
    inputs, is_training: bool, *, module_fun: tp.Callable[[], hk.Module], **kwargs
):
    # graph, node_features = inputs
    coords, data, node_features = inputs
    size = node_features.shape[0]
    graph = spax.COO(coords, data, (size, size))
    module = module_fun(**kwargs)
    return module(graph, node_features, is_training)


@configurable
def fit_semi_supervised_single(
    rng: huf.types.PRNGKey,
    model: huf.models.Model,
    data: data.SemiSupervisedSingle,
    steps: int,
    initial_step: int = 0,
    initial_state: tp.Optional[huf.types.ModelState] = None,
    callbacks: tp.Iterable[huf.callbacks.Callback] = (),
    verbose: bool = True,
):
    # not sure why we need to unpack/repack graph, but avoids errors in GAT
    graph = data.graph.tocoo()
    inputs = (graph.coords, graph.data, data.node_features)
    train_example = inputs, data.labels, data.train_mask.astype(jnp.float32)
    validation_example = (
        inputs,
        data.labels,
        data.validation_mask.astype(jnp.float32),
    )
    test_example = inputs, data.labels, data.test_mask.astype(jnp.float32)

    result = model.fit(
        rng,
        [train_example],
        epochs=steps,
        validation_data=[validation_example],
        initial_state=initial_state,
        initial_epoch=initial_step,
        callbacks=callbacks,
        verbose=verbose,
    )
    model_state = result.model_state
    test_metrics = model.evaluate(
        model_state.params, model_state.net_state, [test_example]
    )
    print(f"Results after {result.epochs} steps")
    for name, metrics in (
        ("train", result.train_metrics),
        ("validation", result.validation_metrics),
        ("test_metrics", test_metrics),
    ):
        print(f"{name} metrics:")
        for k in sorted(metrics):
            print(f"{k}: {metrics[k]}")
    return test_metrics


@configurable
def fit_semi_supervised_single_many(
    rngs: tp.Iterable[huf.types.PRNGKey], *args, **kwargs
):
    if len(rngs) == 0:
        raise ValueError("Must provide at least one rng")
    test_metrics = [
        fit_semi_supervised_single(
            rngs[0], *args, verbose=kwargs.pop("verbose", True), **kwargs
        )
    ]
    test_metrics.extend(
        [
            fit_semi_supervised_single(rng, *args, verbose=False, **kwargs)
            for rng in rngs[1:]
        ]
    )
    test_metrics = jax.tree_util.tree_multimap(lambda *x: jnp.asarray(x), *test_metrics)
    print("---------------------")
    print(f"Metrics over {len(rngs)} runs")
    print("---------------------")
    for k in sorted(test_metrics):
        m = test_metrics[k]
        print(f"{k}: {m.mean()} +- {m.std()}, [{m.min()}, {m.max()}]")
    return test_metrics
