import functools
import typing as tp

import gin
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse_ops import JAXSparse

import haiku as hk
from haiku._src import utils
from huf.module_ops import Linear, dropout
from huf.types import Activation

configurable = functools.partial(gin.configurable, module="grax.hk_utils")


class Renormalize(hk.Module):
    def __init__(
        self,
        create_scale: bool = True,
        create_offset: bool = True,
        name: tp.Optional[str] = None,
    ):
        super().__init__(name=name)
        self.create_scale = create_scale
        self.create_offset = create_offset

    def __call__(self, x):
        assert x.ndim == 2
        size = x.shape[-1]
        if self.create_scale:
            scale = hk.get_parameter(
                "scale", shape=(size,), dtype=x.dtype, init=jnp.ones
            )
            x = x * scale
        if self.create_offset:
            offset = hk.get_parameter(
                "offset", shape=(size,), dtype=x.dtype, init=jnp.zeros
            )
            x = x + offset
        return x


class GatheredBatchNorm(hk.Module):
    """Normalizes inputs to maintain a mean of ~0 and stddev of ~1.

    See: https://arxiv.org/abs/1502.03167.

    There are many different variations for how users want to manage scale and
    offset if they require them at all. These are:

      - No scale/offset in which case ``create_*`` should be set to ``False`` and
        ``scale``/``offset`` aren't passed when the module is called.
      - Trainable scale/offset in which case ``create_*`` should be set to
        ``True`` and again ``scale``/``offset`` aren't passed when the module is
        called. In this case this module creates and owns the ``scale``/``offset``
        variables.
      - Externally generated ``scale``/``offset``, such as for conditional
        normalization, in which case ``create_*`` should be set to ``False`` and
        then the values fed in at call time.

    NOTE: ``jax.vmap(hk.transform(BatchNorm))`` will update summary statistics and
    normalize values on a per-batch basis; we currently do *not* support
    normalizing across a batch axis introduced by vmap.
    """

    def __init__(
        self,
        create_scale: bool,
        create_offset: bool,
        decay_rate: float,
        eps: float = 1e-5,
        scale_init: tp.Optional[hk.initializers.Initializer] = None,
        offset_init: tp.Optional[hk.initializers.Initializer] = None,
        axis: tp.Optional[tp.Sequence[int]] = None,
        cross_replica_axis: tp.Optional[str] = None,
        cross_replica_axis_index_groups: tp.Optional[
            tp.Sequence[tp.Sequence[int]]
        ] = None,
        data_format: str = "channels_last",
        name: tp.Optional[str] = None,
    ):
        """Constructs a BatchNorm module.

        Args:
          create_scale: Whether to include a trainable scaling factor.
          create_offset: Whether to include a trainable offset.
          decay_rate: Decay rate for EMA.
          eps: Small epsilon to avoid division by zero variance. Defaults ``1e-5``,
            as in the paper and Sonnet.
          scale_init: Optional initializer for gain (aka scale). Can only be set
            if ``create_scale=True``. By default, ``1``.
          offset_init: Optional initializer for bias (aka offset). Can only be set
            if ``create_offset=True``. By default, ``0``.
          axis: Which axes to reduce over. The default (``None``) signifies that all
            but the channel axis should be normalized. Otherwise this is a list of
            axis indices which will have normalization statistics calculated.
          cross_replica_axis: If not ``None``, it should be a string representing
            the axis name over which this module is being run within a ``jax.pmap``.
            Supplying this argument means that batch statistics are calculated
            across all replicas on that axis.
          cross_replica_axis_index_groups: Specifies how devices are grouped.
          data_format: The data format of the input. Can be either
            ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
            default it is ``channels_last``.
          name: The module name.
        """
        super().__init__(name=name)
        if not create_scale and scale_init is not None:
            raise ValueError("Cannot set `scale_init` if `create_scale=False`")
        if not create_offset and offset_init is not None:
            raise ValueError("Cannot set `offset_init` if `create_offset=False`")
        if cross_replica_axis is None and cross_replica_axis_index_groups is not None:
            raise ValueError(
                "`cross_replica_axis` name must be specified"
                "if `cross_replica_axis_index_groups` are used."
            )

        self.create_scale = create_scale
        self.create_offset = create_offset
        self.eps = eps
        self.scale_init = scale_init or jnp.ones
        self.offset_init = offset_init or jnp.zeros
        self.axis = axis
        self.cross_replica_axis = cross_replica_axis
        self.cross_replica_axis_index_groups = cross_replica_axis_index_groups
        self.channel_index = utils.get_channel_index(data_format)
        self.mean_ema = hk.ExponentialMovingAverage(decay_rate, name="mean_ema")
        self.var_ema = hk.ExponentialMovingAverage(decay_rate, name="var_ema")

    def __call__(
        self,
        inputs: jnp.ndarray,
        ids: jnp.ndarray,
        is_training: bool,
        test_local_stats: bool = False,
        scale: tp.Optional[jnp.ndarray] = None,
        offset: tp.Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Computes the normalized version of the input.

        Args:
          inputs: An array, where the data format is ``[..., C]``.
          is_training: Whether this is during training.
          test_local_stats: Whether local stats are used when is_training=False.
          scale: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the scale applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_scale=True``.
          offset: An array up to n-D. The shape of this tensor must be broadcastable
            to the shape of ``inputs``. This is the offset applied to the normalized
            inputs. This cannot be passed in if the module was constructed with
            ``create_offset=True``.

        Returns:
          The array, normalized across all but the last dimension.
        """
        if self.create_scale and scale is not None:
            raise ValueError("Cannot pass `scale` at call time if `create_scale=True`.")
        if self.create_offset and offset is not None:
            raise ValueError(
                "Cannot pass `offset` at call time if `create_offset=True`."
            )

        channel_index = self.channel_index
        if channel_index < 0:
            channel_index += inputs.ndim

        if self.axis is not None:
            axis = self.axis
        else:
            axis = [i for i in range(inputs.ndim) if i != channel_index]

        if is_training or test_local_stats:
            mean = jnp.mean(inputs[ids], axis, keepdims=True)
            mean_of_squares = jnp.mean(inputs[ids] ** 2, axis, keepdims=True)
            if self.cross_replica_axis:
                mean = jax.lax.pmean(
                    mean,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups,
                )
                mean_of_squares = jax.lax.pmean(
                    mean_of_squares,
                    axis_name=self.cross_replica_axis,
                    axis_index_groups=self.cross_replica_axis_index_groups,
                )
            var = mean_of_squares - mean ** 2
        else:
            mean = self.mean_ema.average
            var = self.var_ema.average

        if is_training:
            self.mean_ema(mean)
            self.var_ema(var)

        w_shape = [1 if i in axis else inputs.shape[i] for i in range(inputs.ndim)]
        w_dtype = inputs.dtype

        if self.create_scale:
            scale = hk.get_parameter("scale", w_shape, w_dtype, self.scale_init)
        elif scale is None:
            scale = np.ones([], dtype=w_dtype)

        if self.create_offset:
            offset = hk.get_parameter("offset", w_shape, w_dtype, self.offset_init)
        elif offset is None:
            offset = np.zeros([], dtype=w_dtype)

        eps = jax.lax.convert_element_type(self.eps, var.dtype)
        inv = scale * jax.lax.rsqrt(var + eps)
        return (inputs - mean) * inv + offset


@configurable
def mlp(
    x: tp.Union[jnp.ndarray, JAXSparse],
    is_training: bool,
    ids=None,
    num_classes: tp.Optional[int] = None,
    hidden_filters: tp.Union[int, tp.Iterable[int]] = 64,
    dropout_rate: float = 0.8,
    use_batch_norm: bool = False,
    use_layer_norm: bool = False,
    use_renormalize: bool = False,
    use_gathered_batch_norm: bool = False,
    activation: Activation = jax.nn.relu,
    final_activation: Activation = lambda x: x,
    input_dropout_rate: tp.Optional[float] = None,
    batch_norm_decay: float = 0.9,
    renorm_scale: bool = True,
):
    assert (
        sum((use_batch_norm, use_layer_norm, use_renormalize, use_gathered_batch_norm))
        <= 1
    )
    if input_dropout_rate is None:
        input_dropout_rate = dropout_rate
    if isinstance(hidden_filters, int):
        hidden_filters = (hidden_filters,)

    x = dropout(x, input_dropout_rate, is_training=is_training)
    for filters in hidden_filters:
        x = Linear(filters)(x)
        if use_batch_norm:
            x = hk.BatchNorm(renorm_scale, True, batch_norm_decay)(x, is_training)
        if use_layer_norm:
            x = hk.LayerNorm(0, renorm_scale, True)(x)
        if use_renormalize:
            x = Renormalize(renorm_scale, True)(x)
        if use_gathered_batch_norm:
            assert ids is not None
            x = GatheredBatchNorm(True, True, batch_norm_decay)(
                x, is_training=is_training, ids=ids
            )
        x = activation(x)
        x = dropout(x, dropout_rate, is_training=is_training)
    if num_classes is not None:
        x = hk.Linear(num_classes)(x)
    return final_activation(x)


@configurable
class MLP(hk.Module):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name)
        self._kwargs = kwargs

    def __call__(self, x, is_training: bool):
        return mlp(x, is_training, **self._kwargs)
