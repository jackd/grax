import typing as tp

import jax.numpy as jnp


class SpectralData(tp.NamedTuple):
    zero_u: jnp.ndarray
    nonzero_u: jnp.ndarray
    nonzero_w: jnp.ndarray
    eigengap: jnp.ndarray


class EigenDecomposition(tp.NamedTuple):
    w: jnp.ndarray  # [m] values
    u: jnp.ndarray  # [n, m] vectors


def get_coefficient_preset(
    name: str, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0
) -> tp.Sequence[tp.Tuple[float, float, float]]:
    """
    Lookup a list of (alpha,beta,gamma) tuples.

    Each describe the spectral filter basis functions given by a preset name. Each basis
    function may consist of a zero-impulse part, a pseudoinverse part, and a high-pass
    part. Each  part may be scaled manually by the alpha, beta, gamma arguments.

    The preset names are as follows:
    * 'single': one basis function combining all three parts
    * 'independent-parts': three independent basis functions, one for each part
    * 'no-zero-impulse': two basis functions, one for pseudoinverse and one for
    high-pass
    * 'no-high-pass': two basis functions, one for zero-impulse and one for
    pseudoinverse
    * 'independent-zero-impulse', two basis functions, one for zero-impulse and one for
    combined pseudoinverse and high-pass
    * 'independent-pseudoinverse', two basis functions, one for pseudoinverse and one
    for combined zero-impulse and high-pass
    * 'independent-high-pass', two basis functions, one for high-pass and one for
    combined zero-impulse and pseudoinverse
    """
    if name == "single":
        return [(alpha, beta, gamma)]
    if name == "independent-parts":
        return [(alpha, 0.0, 0.0), (0.0, beta, 0.0), (0.0, 0.0, gamma)]
    if name == "only-pseudoinverse":
        return [(0.0, beta, 0.0)]
    if name == "no-zero-impulse":
        return [(0.0, beta, 0.0), (0.0, 0.0, gamma)]
    if name == "no-high-pass":
        return [(alpha, 0.0, 0.0), (0.0, beta, 0.0)]
    if name == "independent-zero-impulse":
        return [(alpha, 0.0, 0.0), (0.0, beta, gamma)]
    if name == "independent-pseudoinverse":
        return [(alpha, 0.0, gamma), (0.0, beta, 0.0)]
    if name == "independent-high-pass":
        return [(alpha, beta, 0.0), (0.0, 0.0, gamma)]

    raise ValueError(f"Unknown coefficient preset: {name}")
