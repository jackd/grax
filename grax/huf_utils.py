import os
import socket
import sys
import time
import typing as tp
from collections import defaultdict
from datetime import datetime
from functools import partial

import gin
import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from huf import callbacks as cb
from huf.avals import zeros_like
from huf.callbacks import (
    EpochProgbarLogger,
    EpochVerboseLogger,
    ProgbarLogger,
    VerboseLogger,
)
from huf.data import as_dataset
from huf.metrics import get_combined_metrics
from huf.models import Model
from huf.models import profile_memory as _profile_memory
from huf.types import FitResult, FitState, Metrics, PRNGKey, Splits

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


class SplitData(tp.NamedTuple):
    train_data: tp.Iterable
    validation_data: tp.Optional[tp.Iterable]
    test_data: tp.Optional[tp.Iterable]


@configurable
def get_split(data: SplitData, split: str = Splits.TRAIN):
    Splits.validate(split)
    return {
        Splits.TRAIN: data.train_data,
        Splits.VALIDATION: data.validation_data,
        Splits.TEST: data.test_data,
    }[split]


def finalize_results(
    model: Model,
    result: FitResult,
    test_data: tp.Optional[tp.Any],
    dt: tp.Optional[float] = None,
    verbose: bool = True,
) -> tp.Mapping[str, jnp.ndarray]:
    if test_data is None:
        test_metrics = None
    else:
        test_metrics = model.evaluate(result.state.model_state, test_data)
    metrics = get_combined_metrics(
        result.train_metrics, result.validation_metrics, test_metrics
    )
    if dt is not None:
        metrics["dt"] = dt
    if verbose:
        max_len = max(len(k) for k in metrics)
        for k in sorted(metrics):
            v = np.asarray(metrics[k])
            if v.size == 1:
                v = v.item()
            print(f"{k.ljust(max_len)} = {v}")
    return metrics


@configurable
def fit_prepared(
    initial_state: tp.Union[int, PRNGKey, FitState],
    model: Model,
    data: SplitData,
    **kwargs,
) -> tp.Mapping[str, jnp.ndarray]:
    t = time.time()
    result = model.fit(
        initial_state, data.train_data, validation_data=data.validation_data, **kwargs
    )
    dt = time.time() - t
    return finalize_results(model, result, data.test_data, dt=dt)


@configurable
def module_call(
    inputs, is_training: bool, *, module_fun: tp.Callable[[], hk.Module], **kwargs
):
    module = module_fun(**kwargs)
    if not isinstance(inputs, (tuple, list)):
        inputs = (inputs,)
    return module(*inputs, is_training)


@configurable
def benchmark_model(model: Model, data: SplitData, seed: int = 0):
    import google_benchmark as benchmark  # pylint: disable=import-outside-toplevel

    train_data = as_dataset(data.train_data).repeat()
    validation_data = as_dataset(data.validation_data).repeat()

    dummy_example = jax.tree_map(zeros_like, train_data.element_spec)
    model.compile(*dummy_example)
    rng = hk.PRNGSequence(seed)
    params, net_state, opt_state = model.init(next(rng), dummy_example[0])
    train_step = model.compiled_train_step
    test_step = model.compiled_test_step
    metrics_state = model.init_metrics_state

    # pylint: disable=expression-not-assigned
    def train_benchmark(state):

        train_iter = iter(train_data)
        example = next(train_iter)

        params_, net_state_, opt_state_, metrics_state_, *_ = train_step(
            params, net_state, next(rng), opt_state, metrics_state, *example
        )

        [x.block_until_ready() for x in jax.tree_flatten(params_)[0]]
        while state:
            params_, net_state_, opt_state_, metrics_state_, *_ = train_step(
                params_, net_state_, next(rng), opt_state_, metrics_state_, *example
            )
            example = next(train_iter)
            [x.block_until_ready() for x in jax.tree_flatten(params_)[0]]

    def test_benchmark(state, data):
        metrics_state_ = metrics_state

        data_iter = iter(data)
        example = next(data_iter)
        metrics_state_, preds, loss, metrics = test_step(
            params, net_state, metrics_state, *example
        )
        [
            x.block_until_ready()
            for x in jax.tree_flatten((metrics_state_, metrics, preds, loss))[0]
        ]
        while state:
            metrics_state_, preds, loss, metrics = test_step(
                params, net_state, metrics_state_, *example
            )
            example = next(data_iter)
            [
                x.block_until_ready()
                for x in jax.tree_flatten((metrics_state_, metrics, preds, loss))[0]
            ]

    # pylint: enable=expression-not-assigned
    benchmark.register(train_benchmark, name="UNTRUSTWORTHY")
    benchmark.register(train_benchmark, name="train_benchmark1")
    benchmark.register(
        partial(test_benchmark, data=validation_data), name="validation_benchmark"
    )
    if data.test_data is not None:
        test_data = as_dataset(data.test_data).repeat()

        benchmark.register(
            partial(test_benchmark, data=test_data), name="test_benchmark"
        )

    benchmark.main(argv=sys.argv[:1])


@configurable
def profile_memory(model: Model, data: SplitData, compiled: bool):
    _profile_memory(model, data.train_data, compiled=compiled)


@configurable
def get_logger(
    freq: str = "epoch",
    progbar: bool = True,
    print_fun: tp.Callable[[str], tp.Any] = print,
):
    assert freq in ("epoch", "step"), freq
    if progbar:
        if freq == "epoch":
            return EpochProgbarLogger(print_fun)
        return ProgbarLogger(print_fun)
    if freq == "epoch":
        return EpochVerboseLogger(print_fun)
    return VerboseLogger(print_fun)


@configurable
def get_fit_callbacks(
    logger: tp.Optional[cb.Callback] = None,
    early_stopping: tp.Optional[cb.Callback] = None,
    tb_dir: tp.Optional[str] = None,
    terminate_on_nan: bool = True,
) -> tp.Sequence[cb.Callback]:
    cbs = [c for c in (logger, early_stopping) if c is not None]
    if tb_dir is not None:
        cbs.append(cb.TensorBoard(tb_dir))
    if terminate_on_nan:
        cbs.append(cb.TerminateOnNaN())
    return cbs


@configurable
def get_logdir(base_dir: str, name: tp.Optional[str] = None):
    if name is None:
        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        name = current_time + "_" + socket.gethostname()
    return os.path.join(base_dir, name)
