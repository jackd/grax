import functools

import gin

configurable = functools.partial(gin.configurable, module="dagnn.utils")


@configurable
def one_minus(x):
    return 1 - x
