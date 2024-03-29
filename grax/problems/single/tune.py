import typing as tp
from collections import defaultdict
from functools import partial

import gin
import numpy as np
from huf.objectives import DEFAULT_OBJECTIVE, Objective
from huf.ray.tune.utils import full_metric_name, get_results
from huf.types import Modes, Splits
from ray import tune

configurable = partial(gin.configurable, module="grax.problems.single.tune")


def get_best_trial(analysis: tune.ExperimentAnalysis, objective: Objective, scope: str):
    return analysis.get_best_trial(
        full_metric_name(objective), mode=objective.mode, scope=scope
    )


def get_best_trials(
    analysis: tune.ExperimentAnalysis,
    objective: Objective,
    scope: str,
    flatten_keys=("seed",),
):
    def items(x):
        if hasattr(x, "item"):
            if x.size == 1:
                return x.item()
            return tuple(el.item() for el in x.flatten())
        return x

    grouped_trials = defaultdict(list)
    for trial in analysis.trials:
        keys = sorted((k for k in trial.config if k not in flatten_keys))
        grouped_trials[tuple(items(trial.config[k]) for k in keys)].append(trial)

    metric = full_metric_name(objective)
    return Modes.reducer(objective.mode)(
        (
            (trials, np.mean([t.metric_analysis[metric][scope] for t in trials]))
            for trials in grouped_trials.values()
        ),
        key=lambda x: x[1],
    )[0]


@configurable(denylist=("analysis",))
def print_best_config_and_results(
    analysis: tune.ExperimentAnalysis,
    objective: Objective = DEFAULT_OBJECTIVE,
    scope="avg",
    flatten_keys: tp.Optional[tp.Iterable[str]] = None,
    print_fun=print,
):
    combined = defaultdict(list)

    def process_result(result):
        for k, v in result.items():
            if any((k.startswith(f"{m}_") for m in Splits.all())):
                combined[k].append(v)

    if flatten_keys is None:
        best_trial = get_best_trial(analysis, objective, scope)
        result = get_results(best_trial)
        for r in result:
            process_result(r)
        config = best_trial.config
    else:
        trials = get_best_trials(analysis, objective, scope, flatten_keys)
        config = dict(**trials[0].config)
        for k in flatten_keys:
            del config[k]

        for trial in trials:
            result = get_results(trial)
            for r in result:
                process_result(r)

    lines = ["Best config:"]
    for k in sorted(config):
        lines.append(f"{k} = {config[k]}")
    lines.append("Results:")
    key_len = max(len(k) for k in combined)
    for k in sorted(combined):
        v = combined[k]
        mean = np.mean(v)
        std = np.std(v)
        lines.append(f"{k.ljust(key_len)} = {mean} +- {std}")
    print_fun("\n".join(lines))


# @configurable
# def with_best_trials(
#     analysis: tune.ExperimentAnalysis,
#     fun: tp.Callable = get_results,
#     objective: Objective = DEFAULT_OBJECTIVE,
#     scope: str = "avg",
#     flatten_keys=("seed",),
# ):
#     best_trials = get_best_trials(
#         analysis, objective=objective, scope=scope, flatten_keys=flatten_keys
#     )
#     results = []
#     metric = full_metric_name(objective)
#     for trial in best_trials:
#         checkpoint = analysis.get_best_checkpoint(trial, metric, mode=objective.mode)
#         results.append(fun(trial, checkpoint))
#     return results
