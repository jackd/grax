import functools
import os
import typing as tp

import gin
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import spax
from jax.experimental.sparse import COO

from grax.graph_utils.transforms import symmetric_normalize
from grax.problems.single.data import SemiSupervisedSingle, SplitData
from grax.projects.pigcn import utils

configurable = functools.partial(gin.configurable, module="pigcn.data")


def get_spectral_data(
    w: jnp.ndarray,
    u: jnp.ndarray,
    threshold: float = 1e-2,
    max_rank: tp.Optional[int] = None,
) -> utils.SpectralData:
    w = jnp.asarray(w)
    u = jnp.asarray(u)
    nonzero_mask = w > threshold

    zero_u = u[:, ~nonzero_mask]
    nonzero_u = u[:, nonzero_mask]
    nonzero_w = w[nonzero_mask]
    eigengap = nonzero_w.min() if nonzero_mask.sum() > 0 else jnp.zeros((), w.dtype)

    ind = jnp.argsort(nonzero_w)
    nonzero_w = nonzero_w[ind]
    nonzero_u = nonzero_u[:, ind]

    if max_rank is not None and nonzero_w.size > max_rank:
        nonzero_w = nonzero_w[:max_rank]
        nonzero_u = nonzero_u[:, :max_rank]

    return utils.SpectralData(zero_u, nonzero_u, nonzero_w, eigengap)


def precompute_u0_zero(row, col):
    raise NotImplementedError("TODO")


def graph_laplacian_decomposition(
    adj: COO,
    num_ev: tp.Optional[int] = None,
    tol: tp.Union[int, float] = 0,
    precomputed_u0: tp.Optional[jnp.ndarray] = None,
) -> utils.EigenDecomposition:
    """
    Get a (partial) eigen decomposition of the graph Laplacian.

    If `num_ev` is not `None`, only that many smallest eigenvalues are computed. The
    parameter tol is used for scipy.linalg.eigs (if it is called).
    """
    n = adj.shape[0]
    adj = sp.coo_matrix((adj.data, (adj.row, adj.col)), shape=adj.shape).tocsr()
    if num_ev is None or num_ev > n / 2:
        if sp.issparse(adj):
            adj = adj.toarray()
        w, U = np.linalg.eigh(adj)
        w = 1 - w
        ind = np.argsort(w)
        if num_ev is not None:
            ind = ind[:num_ev]
        w = w[ind]
        U = U[:, ind]
    else:
        if precomputed_u0 is not None:
            matvec = lambda x: adj @ x + x - 2 * precomputed_u0 @ (precomputed_u0.T @ x)
            shifted_adj = sp.linalg.LinearOperator((n, n), matvec)
            num_ev -= precomputed_u0.shape[1]
        elif sp.issparse(adj):
            shifted_adj = (adj + sp.identity(adj.shape[0])).tocsr()
        else:
            shifted_adj = adj + np.identity(adj.shape[0])

        np.random.seed(0)
        v0 = (
            np.random.default_rng(0)
            .normal(loc=1.0, scale=0.1, size=(n,))
            .astype(shifted_adj.dtype)
        )
        np.random.seed(0)
        w, u = sp.linalg.eigsh(shifted_adj, num_ev, tol=tol, v0=v0)
        # np.random.seed(0)
        # w_, u_ = sp.linalg.eigsh(shifted_adj, num_ev, tol=tol, v0=v0)
        w = 2 - w

        if precomputed_u0 is not None:
            u = np.hstack((precomputed_u0, u))
            w = np.hstack((np.zeros(precomputed_u0.shape[1]), w))
    return utils.EigenDecomposition(
        jnp.asarray(w.astype(np.float32)), jnp.asarray(u.astype(np.float32))
    )


def preconvolve_input(spectral_data: utils.SpectralData, features: jnp.ndarray, coeffs):
    """
    Precompute the convolutions of the given input matrix with the model's
    basis function. The result can be passed to the network's forward
    operation.
    """
    X = spax.ops.to_dense(features)  # just in case
    zero_u, nonzero_u, nonzero_w, eigengap = spectral_data
    del spectral_data, features

    # zero_U_X = zero_u.T @ X
    # nonzero_U_X = nonzero_u.T @ X

    # result = []
    # for alpha, beta, gamma in coeffs:
    #     s = 0
    #     if alpha != 0 or gamma != 0:
    #         s += (alpha - gamma * eigengap) * (zero_u @ zero_U_X)
    #     if beta != 0 or gamma != 0:
    #         s += nonzero_u @ (
    #             eigengap * (beta / nonzero_w[:, np.newaxis] - gamma) * nonzero_U_X
    #         )
    #     if gamma != 0:
    #         s += eigengap * gamma * X
    #     result.append(s)
    # return result

    zero_U_X = zero_u.T @ X
    nonzero_U_X = nonzero_u.T @ X

    result = []
    for alpha, beta, gamma in coeffs:
        s_terms = []
        if alpha or gamma:
            s_terms.append((alpha - gamma * eigengap) * (zero_u @ zero_U_X))
        if beta or gamma:
            s_terms.append(
                nonzero_u
                @ (
                    eigengap
                    * (beta / jnp.expand_dims(nonzero_w, 1) - gamma)
                    * nonzero_U_X
                )
            )
        if gamma:
            s_terms.append(eigengap * gamma * X)
        result.append(sum(s_terms))
    return result


@configurable
def preprocess_inputs(
    data: SemiSupervisedSingle,
    rank: int,
    coeffs="independent-parts",
    eig_tol: tp.Union[float, int] = 0,
    eig_threshold: float = 1e-6,
    precompute_u0: bool = False,
    override_path: tp.Optional[str] = None,
):
    if isinstance(coeffs, str):
        coeffs = utils.get_coefficient_preset(coeffs)
    adj = symmetric_normalize(data.graph)
    if precompute_u0:
        u0 = precompute_u0_zero(adj.row, adj.col)
        num_ev = rank + u0.shape[1]
    else:
        u0 = None
        num_ev = rank + 1

    w, u = graph_laplacian_decomposition(  # pylint: disable=unpacking-non-sequence
        adj, num_ev, tol=eig_tol, precomputed_u0=u0
    )

    if override_path is not None:
        override_path = os.path.expanduser(os.path.expandvars(override_path))
        override_data = np.load(override_path)
        override_data = {k: jnp.asarray(v) for k, v in override_data.items()}
        features = override_data.pop("preconvolved_input")
        spectral_data = utils.SpectralData(**override_data)
    else:
        spectral_data = get_spectral_data(w, u, eig_threshold, max_rank=rank)
        features = preconvolve_input(spectral_data, data.node_features, coeffs)

    # # HACK
    # hacked_data = hacked_inputs()
    # spectral_data, features, _ = hacked_data[0][0][0]

    # r = min(spectral_data.nonzero_w.size, spectral_data_.nonzero_w.size)
    # print(
    #     np.stack(
    #         (spectral_data.nonzero_w.to_py()[:r], spectral_data_.nonzero_w.to_py()[:r]),
    #         axis=1,
    #     )
    # )
    # print(spectral_data.nonzero_w.size, spectral_data_.nonzero_w.size)
    # print(spectral_data.zero_u.shape, spectral_data_.zero_u.shape)
    # exit()

    features = jnp.concatenate(features, axis=-1)

    def get_example(ids):
        return (spectral_data, features, ids), data.labels[ids]

    return SplitData(
        (get_example(data.train_ids),),
        (get_example(data.validation_ids),),
        (get_example(data.test_ids),),
    )


@configurable
def hacked_inputs(path: str) -> SplitData:
    data = np.load(path)
    (
        zero_U,
        nonzero_U,
        nonzero_w,
        eigengap,
        preconvolved_input,
        train_mask,
        validation_mask,
        test_mask,
        labels,
    ) = (
        jnp.asarray(data[k])
        for k in (
            "zero_U",
            "nonzero_U",
            "nonzero_w",
            "eigengap",
            "preconvolved_input",
            "train_mask",
            "val_mask",
            "test_mask",
            "y",
        )
    )
    spectral_data = utils.SpectralData(zero_U, nonzero_U, nonzero_w, eigengap)
    # features = [f for f in preconvolved_input]
    features = jnp.concatenate(preconvolved_input, axis=-1)
    train_ids, validation_ids, test_ids = (
        jnp.where(m) for m in (train_mask, validation_mask, test_mask)
    )

    def get_example(ids):
        return (spectral_data, features, ids), labels[ids]

    return SplitData(
        (get_example(train_ids),),
        (get_example(validation_ids),),
        (get_example(test_ids),),
    )
