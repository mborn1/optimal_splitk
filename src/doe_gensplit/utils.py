import numba
import numpy as np
from numba.typed import List

CACHE = False

##################################################################
##  NUMBA COMPATIBLE NUMPY FUNCTIONS
##################################################################

@numba.njit(cache=CACHE)
def np_all_axis1(x):
    """
    Numba compatible implementation of np.all(..., axis=1) for
    2d arrays

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    x : np.array(2d)
        The input array

    Returns
    -------
    out : np.array(1d, bool)
        The results of np.all(..., axis=1)
    """
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[0]):
        out[i] = np.all(x[i, :])
    return out

@numba.njit(cache=CACHE)
def np_take_advanced(arr, idx, out=None):
    """
    Take elements from a NumPy array using advanced indexing as Numba is
    not capable.
    The indices should be values as if arr is a 1d array.

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    arr : np.array(2d)
        The array to take elements from
    idx : np.array
        The indices (can be an n-d array)
    out : np.array
        Same shape is idx, possible to use preallocation
    
    Returns
    -------
    out : np.array
        The elements from arr in the shape of idx.
    """
    # Reshape indices
    shape = idx.shape
    idx = idx.flatten()

    # Initialize out
    if out is None:
        out = np.zeros((idx.size, *arr.shape[1:]), dtype=arr.dtype)

    # Fill result
    for i in range(idx.size):
        out[i] = arr[idx[i]]

    # Reshape and return
    return out.reshape((*shape, *arr.shape[1:]))

@numba.njit(cache=CACHE)
def np_argmax1(arr):
    out = np.zeros(arr.shape[0], dtype=np.int64)
    for i in range(out.size):
        out[i] = np.argmax(arr[i])
    return out

##################################################################
##  GENERAL UTILS
##################################################################

@numba.njit(cache=CACHE)
def obs_var(plot_sizes, alphas=None, betas=None, ratios=None):
    """
    The observation variance matrix of the generalized split-plot model.

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The array of plot sizes according to the generalized split-plot model.
    alphas : np.array(1d)
        The alpha values, possibly precomputed
    betas : np.array(1d)
        The beta values, possibly precomputed
    ratios : np.array(1d)
        The ratios of the variance components to $\sigma^2_1$. If not
        provided, they are all 1, if provided, the first element should be
        1 and the size should be equal to that of the plot sizes.

    Returns
    -------
    V : np.array(2d)
        The observation variance matrix
    """
    # Initialize alphas and betas
    if alphas is None:
        alphas = np.cumprod(plot_sizes[::-1])[::-1]
    if betas is None:
        betas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))
    if ratios is None:
        ratios = np.ones_like(plot_sizes, dtype=np.float64)

    # Compute variance-covariance of observations
    V = np.zeros((alphas[0], alphas[0]))
    for i in range(plot_sizes.size):
        Zi = np.kron(np.eye(alphas[i]), np.ones((betas[i], 1)))
        V += ratios[i] * Zi @ Zi.T

    return V

