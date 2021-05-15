import time
import typing as tp
from collections import defaultdict
from functools import partial

import gin
import numpy as np

import haiku as hk
import huf
import jax
from huf.experiments import ExperimentCallback, experiment_context
from huf.metrics import get_combined_metrics
from huf.models import Model
from huf.types import FitResult, FitState, Metrics, PRNGKey

configurable = partial(gin.configurable, module="grax.huf_utils")


@configurable
def print_moments(
    results: tp.Sequence[Metrics],
    print_fun: tp.Callable[[str], None] = print,
    skip: int = 0,
):
    out = defaultdict(list)
    for r in results[skip:]:
        for k, v in r.items():
            out[k].append(v)
    max_len = max(len(k) for k in out)
    for k in sorted(out):
        v = out[k]
        print_fun(f"{k.ljust(max_len)} = {np.mean(v):.5f} +- {np.std(v):.5f}")


class FitData(tp.NamedTuple):
    model: Model
    train_data: tp.Iterable
    validation_data: tp.Iterable
    test_data: tp.Optional[tp.Iterable]


@configurable
def fit_prepared(
    data: FitData, initial_state: tp.Union[int, PRNGKey, FitState], **kwargs,
) -> tp.Union[FitResult, tp.Tuple[FitResult, Metrics]]:
    result = data.model.fit(
        initial_state, data.train_data, validation_data=data.validation_data, **kwargs
    )
    if data.test_data is None:
        test_metrics = None
    else:
        test_metrics = data.model.evaluate(result.state.model_state, data.test_data)
    return get_combined_metrics(
        result.train_metrics, result.validation_metrics, test_metrics
    )


@configurable
def fit_many(
    data: FitData,
    num_repeats: int,
    rng: tp.Union[int, huf.types.PRNGKey],
    experiment_callbacks: tp.Iterable[ExperimentCallback] = (),
    **kwargs,
) -> tp.List[Metrics]:
    assert isinstance(rng, (int, huf.types.PRNGKey))
    if isinstance(rng, int):
        rng = jax.random.PRNGKey(rng)

    results = []
    for state in jax.random.split(rng, num_repeats):
        with experiment_context(experiment_callbacks):
            t = time.time()
            result = fit_prepared(data, initial_state=state, **kwargs)
            result["time"] = time.time() - t
        for callback in experiment_callbacks:
            callback.on_done(result)
        results.append(result)

    return results


@configurable
def module_call(
    inputs, is_training: bool, *, module_fun: tp.Callable[[], hk.Module], **kwargs
):
    module = module_fun(**kwargs)
    if isinstance(inputs, tuple):
        return module(*inputs, is_training)
    return module(inputs, is_training)
