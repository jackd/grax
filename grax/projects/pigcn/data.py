import functools
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


def symmetric_normalize_numpy(adj):
    # numerically different to grax.graph_utils.transofrms.symmetric_normalize
    if not sp.issparse(adj):
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)), shape=adj.shape)
    d = np.squeeze(np.asarray(adj.sum(1)))
    d = np.where(d == 0, np.zeros_like(d), 1 / np.sqrt(d))
    data = adj.data * d[adj.row] * d[adj.col]
    return sp.coo_matrix((data, (adj.row, adj.col)), shape=adj.shape)


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
    if not sp.issparse(adj):
        adj = sp.coo_matrix((adj.data, (adj.row, adj.col)), shape=adj.shape)
    n = adj.shape[0]
    adj = adj.tocsr()
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
        v0 = np.random.uniform(size=(n,)).astype(shifted_adj.dtype)
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
):
    if isinstance(coeffs, str):
        coeffs = utils.get_coefficient_preset(coeffs)
    # adj = symmetric_normalize_numpy(data.graph)
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
    spectral_data = get_spectral_data(w, u, eig_threshold, max_rank=rank)
    features = data.node_features
    # undo dgl's automatic row-normalization
    if isinstance(features, jnp.ndarray):
        features = jnp.where(features == 0, features, jnp.ones_like(features))
    else:
        features = spax.ops.map_data(features, jnp.ones_like)
    features = preconvolve_input(spectral_data, features, coeffs)
    features = jnp.concatenate(features, axis=-1)

    def get_example(ids):
        return (spectral_data, features, ids), data.labels[ids]

    return SplitData(
        (get_example(data.train_ids),),
        (get_example(data.validation_ids),),
        (get_example(data.test_ids),),
    )
