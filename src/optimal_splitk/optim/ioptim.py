from ..optimizers import compute_update, inv_update_no_P, Optim
from ..utils import obs_var, CACHE
from ..init import initialize
from ..encode import encode_design
from ..doe import x2fx
from collections import namedtuple
import numba
import numpy as np

IoptimPreState = namedtuple('IoptimPreState', 'plot_sizes betas betas_inv c V moments')
IoptimState = namedtuple('IoptimState', 'plot_sizes betas betas_inv c V moments Minv metric')

@numba.njit(cache=CACHE)
def outer_integral(arr):
    """
    Computes the integral of the outer products of the array rows (simple average)

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    arr : np.array(2d)
        The array
    
    Returns
    -------
    out : np.array(2d)
        The integral of the outer product
    """
    out = np.zeros((arr.shape[-1], arr.shape[-1]))
    for i in range(arr.shape[0]):
        out += arr[i].T @ arr[i]
    return out / arr.shape[0]

@numba.njit(cache=CACHE)
def preinit(plot_sizes, model, factors, ratios):
    """
    Pre-initialize some constants necessary for computing metric updates.

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    plot_sizes : np.array(1d)
        The plot sizes corresponding to the generalized split-plot model
    model : np.array(2d)
        The model in Matlab notation (each row is a term, elements specify the powers, columns the effects)
    factors : np.array(2d)
        The array specifying the factor types, each row is a term
        with the level and 1 if continuous or larger for categorical.
    ratios : np.array(1d)
        The variance ratios to the first (run) variance
    
    Returns
    -------
    pre_state : :py:class:`IoptimPreState`
        The pre state
    """
    # Alphas and betas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]
    betas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))

    # Betas inverse
    betas_inv = np.cumsum(np.concatenate((np.array([0], dtype=np.float64), 1/betas[1:])))

    # Compute c-coefficients
    c = np.zeros(plot_sizes.size)
    c[0] = 1
    for i in range(1, c.size):
        c[i] = -ratios[i] * np.sum(betas[:i] * c[:i]) / np.sum(ratios[:i+1] * betas[:i+1])

    # Compute information matrix
    V = obs_var(plot_sizes, alphas=alphas, betas=betas, ratios=ratios)

    # Compute moments matrix
    samples = initialize(np.ones_like(plot_sizes), factors, n=10000)
    samples = x2fx(encode_design(samples, factors), model[1])
    moments = outer_integral(samples)

    return IoptimPreState(plot_sizes, betas, betas_inv, c, V, moments)

@numba.njit(cache=CACHE)
def init(prestate, Y, X): 
    """
    I-optimal criterion: initialization (try specific).

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    prestate : :py:class:`IoptimPreState`
        The prestate returned by :py:func:`preinit`
    Y : np.array(2d)
        Not used, for compatibility purposes
    X : np.array(2d)
        The design matrix

    Returns
    -------
    state : :py:class:`IoptimState`
        Return a state object
    """   
    # Validate rank of matrix
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise np.linalg.LinAlgError('Matrix M is singular')

    # Compute information matrix
    M = X.T @ np.linalg.solve(prestate.V, X)

    # Invert information matrix
    Minv = np.linalg.inv(M)

    # Compute the initial metric
    metric = np.array([__metric(prestate.moments, Minv)])

    return IoptimState(prestate.plot_sizes, prestate.betas, prestate.betas_inv, prestate.c, prestate.V, prestate.moments, Minv, metric)

@numba.njit(cache=CACHE)
def __metric(moments, Minv):
    """
    Computes the metrics from the moments matrix and the inverse information
    matrix

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    moments : np.array(2d)
        The moments matrix
    Minv : np.array(2d)
        The inverse information matrix
    
    Returns
    -------
    metric : float
        The I-optimality metric of the current design
    """
    # Compute trace(Mx^{-1} @ M) (average prediction variance)
    # Minus sign for minimization
    metric = -np.sum(Minv * moments.T)
    return metric

@numba.njit(cache=CACHE)
def metric(state, Y, X):
    """
    I-optimal criterion: metric

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    prestate : :py:class:`IoptimPreState`
        The prestate returned by :py:func:`preinit`
    Y : np.array(2d)
        Not used, for compatibility purposes
    X : np.array(2d)
        Not used, for compatibility purposes

    Returns
    -------
    state : :py:class:`IoptimState`
        Return a state object
    """
    # Doesn't use Y and X as the inverse of the information matrix 
    # is already in the state from the update
    return __metric(state.moments, state.Minv)

@numba.njit(cache=CACHE)
def update(state, X, Xi_star, level, grp):
    """
    I-optimal criterion: update formula

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    state : :py:class:`IoptimState`
        The state object of the optimizer
    X : np.array(2d)
        The model matrix of the current design
    Xi_star : np.array(2d)
        The new block of rows
    level : int
        The split-level of the update
    grp : int
        The group number within the given split-level that was updated

    Returns
    -------
    improved : bool
        Whether the metric was improved
    state : :py:class:`IoptimState`
        The new state
    """
    # Compute U,D from coordinate exchange update
    U, D = compute_update(level, grp, X, Xi_star, state.plot_sizes, state.c, betas=state.betas, betas_inv=state.betas_inv)

    # Compute change in inverse
    Minv_u = inv_update_no_P(U, D, state.Minv)

    # Compute update to the metric (!minus sign!)
    i_update = -np.sum(Minv_u * state.moments.T)

    # Minimize prediction variance 
    # (numerical instability may cause variance to become negative otherwise)
    if i_update < 0 and i_update >= state.metric[0]:
        # Update inv(M)
        Minv = state.Minv
        Minv -= Minv_u

        # Update the metric
        metric = state.metric
        metric -= i_update

        return True, state
    
    # Return no update
    return False, state

# Create optimizer
Ioptim = Optim(preinit, init, update, metric)
