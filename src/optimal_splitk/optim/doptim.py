from ..optimizers import compute_update, det_update, inv_update, Optim
from ..utils import obs_var, CACHE
from collections import namedtuple
import numba
import numpy as np

# The prestate structure
DoptimPreState = namedtuple('DoptimPreState', 'plot_sizes alphas betas betas_inv c V')

# The state of the optimizer
DoptimState = namedtuple('DoptimState', 'plot_sizes alphas betas betas_inv c V Minv')

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
        Not used, for interface compatibility
    factors : np.array(2d)
        Not used, for interface compatibility
    ratios : np.array(1d)
        The variance ratios to the first (run) variance
    
    Returns
    -------
    pre_state : :py:class:`DoptimPreState`
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

    return DoptimPreState(plot_sizes, alphas, betas, betas_inv, c, V)

@numba.njit(cache=CACHE)
def init(prestate, Y, X): 
    """
    D-optimal criterion: initialization (try specific).

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    prestate : :py:class:`DoptimPreState`
        The prestate returned by :py:func:`preinit`
    Y : np.array(2d)
        Not used, for compatibility purposes
    X : np.array(2d)
        The design matrix

    Returns
    -------
    state : :py:class:`DoptimState`
        Return a state object
    """   
    # Compute information matrix
    M = X.T @ np.linalg.solve(prestate.V, X)

    # Invert information matrix
    Minv = np.linalg.inv(M)

    return DoptimState(prestate.plot_sizes, prestate.alphas, prestate.betas, prestate.betas_inv, prestate.c, prestate.V, Minv)

@numba.njit(cache=CACHE)
def metric(state, Y, X):
    """
    D-optimal criterion: metric

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    prestate : :py:class:`DoptimPreState`
        The prestate returned by :py:func:`preinit`
    Y : np.array(2d)
        Not used, for compatibility purposes
    X : np.array(2d)
        The design matrix

    Returns
    -------
    state : :py:class:`DoptimState`
        Return a state object
    """
    # Compute determinant of information matrix
    metric = np.linalg.det(X.T @ np.linalg.solve(state.V, X))
    return metric

@numba.njit(cache=CACHE)
def update(state, X, Xi_star, level, grp):
    """
    D-optimal criterion: update formula

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    state : :py:class:`DoptimState`
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
    state : :py:class:`DoptimState`
        The new state
    """
    # Compute U,D from coordinate exchange update
    U, D = compute_update(level, grp, X, Xi_star, state.plot_sizes, state.c, betas=state.betas, betas_inv=state.betas_inv)

    # Compute change in determinant
    du, P = det_update(U, D, state.Minv)

    if du > 1:
        # Update inv(M)
        Minv = state.Minv
        Minv -= inv_update(U, D, state.Minv, P=P)
        return True, state
    
    # Return no update
    return False, state

# Create optimizer
Doptim = Optim(preinit, init, update, metric)
