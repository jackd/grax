import functools
import typing as tp

import gin
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from huf.initializers import glorot_uniform

from grax import hk_utils, optax_utils
from grax.projects.pigcn import utils

configurable = functools.partial(gin.configurable, module="pigcn")


class PseudoInverseGraphConvolution(hk.Module):
    def __init__(
        self,
        filters: int,
        coeffs: tp.Union[tp.Sequence[tp.Tuple[float, float, float]], str],
        with_bias: bool = True,
        b_init: tp.Callable = jnp.zeros,
        w_init: tp.Callable = glorot_uniform,
        name=None,
    ):
        super().__init__(name=name)
        if isinstance(coeffs, str):
            coeffs = utils.get_coefficient_preset(coeffs)
        self.filters = filters
        self.coeffs = coeffs
        self.with_bias = with_bias
        self.b_init = b_init
        self.w_init = w_init

    def __call__(
        self,
        spectral_data: utils.SpectralData,
        features: jnp.ndarray,
        ids: tp.Optional[jnp.ndarray] = None,
    ):
        zero_terms = []
        nonzero_terms = []
        terms = []

        def masked(x):
            if ids is None:
                return x
            return x[ids]

        weights = hk.get_parameter(
            "w",
            shape=(len(self.coeffs), features.shape[-1], self.filters),
            dtype=features.dtype,
            init=self.w_init,
        )

        X = features
        for (alpha, beta, gamma), w in zip(self.coeffs, weights):
            Y = X @ w
            if alpha or gamma:
                zero_terms.append(
                    (alpha - spectral_data.eigengap * gamma)
                    * (spectral_data.zero_u.T @ Y)
                )
            if beta or gamma:
                nonzero_terms.append(
                    (beta / jnp.expand_dims(spectral_data.nonzero_w, 1) - gamma)
                    * spectral_data.eigengap
                    * (spectral_data.nonzero_u.T @ Y)
                )
            if gamma:
                terms.append(spectral_data.eigengap * gamma * masked(Y))

        if zero_terms:
            terms.append(masked(spectral_data.zero_u) @ sum(zero_terms))
        if nonzero_terms:
            terms.append(masked(spectral_data.nonzero_u) @ sum(nonzero_terms))

        if self.with_bias is not None:
            b = hk.get_parameter(
                name="b", shape=(self.filters,), dtype=features.dtype, init=self.b_init,
            )
            terms.append(b)
        return sum(terms)


@configurable
def partitioned_additive_weight_decay(weight_decay: float):
    def predicate(layer_name, param_name, value):
        del layer_name, value
        return param_name == "w"

    return optax_utils.partition(predicate, optax.additive_weight_decay(weight_decay))


@configurable
class PIGCN(hk.Module):
    def __init__(
        self,
        num_classes: int,
        coeffs="independent-parts",
        hidden_filters: tp.Sequence[int] = (32,),
        dropout_rate: float = 0.5,
        activation: tp.Callable = jax.nn.relu,
        name=None,
    ):
        super().__init__(name=name)
        if isinstance(coeffs, str):
            coeffs = utils.get_coefficient_preset(coeffs)
        self.num_classes = num_classes
        self.coeffs = coeffs
        if isinstance(hidden_filters, int):
            hidden_filters = (hidden_filters,)
        else:
            self.hidden_filters = tuple(hidden_filters)
        self.dropout_rate = dropout_rate
        self.activation = activation

    def __call__(
        self,
        spectral_data: utils.SpectralData,
        features,
        ids: jnp.ndarray,
        is_training: bool = False,
    ) -> jnp.ndarray:
        def activate(x):
            x = self.activation(x)
            return hk_utils.dropout(x, rate=self.dropout_rate, is_training=is_training)

        features = hk.Linear(self.hidden_filters[0])(features)

        for h in self.hidden_filters[1:]:
            features = activate(features)
            features = PseudoInverseGraphConvolution(h, self.coeffs)(
                spectral_data, features,
            )
        features = activate(features)
        features = PseudoInverseGraphConvolution(self.num_classes, self.coeffs)(
            spectral_data, features, ids
        )
        return features
