import functools

import gin
import spax

from grax.graph_utils.transforms import to_format
from grax.problems.single.data import SemiSupervisedSingle, SplitData

configurable = functools.partial(gin.configurable, module="sgc.data")


@configurable
def preprocess_inputs(data: SemiSupervisedSingle, degree: int = 2, fmt: str = "dense"):
    A = data.graph
    features = spax.ops.to_dense(data.node_features)
    for _ in range(degree):
        features = A @ features
    features = to_format(features, fmt)

    def get_dataset(ids):
        return ((features[ids], data.labels[ids]),)

    return SplitData(
        *(
            get_dataset(ids)
            for ids in (data.train_ids, data.validation_ids, data.test_ids)
        )
    )
