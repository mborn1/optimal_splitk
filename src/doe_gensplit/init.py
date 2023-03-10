import numpy as np
import numba
from .utils import np_argmax1, CACHE

@numba.njit(cache=CACHE)
def __init_unconstrained(factors, Y, alphas, betas, coords=None):
    """
    This function is created to avoid possible recursion. Numba has issues with it.

    .. note::
        See :py:func:`initialize_single` for more information

    .. note::
        This function is Numba accelerated

    """
    ##################################################
    # UNCONSTRAINED DESIGN
    ##################################################
    # Loop over all columns
    for col in range(factors.shape[0]):
        # Extract parameters
        level = factors[col, 0]
        typ = factors[col, 1]

        # Generate random values
        n = alphas[level]
        size = betas[level]

        if coords is None:
            if typ == 1:
                # Continuous factor
                r = np.random.rand(n) * 2 - 1
            else:
                # Discrete factor
                choices = np.arange(typ, dtype=np.float64)
                if typ >= n:
                    r = np.random.choice(choices, n, replace=False)
                else:
                    n_replicates = n // choices.size
                    r = np.random.permutation(np.concatenate((np.repeat(choices, n_replicates), np.random.choice(choices, n - choices.size * n_replicates))))
        else:
            # Extract the possible coordinates
            if typ > 1:
                # Convert to decoded values for categorical factors
                c = coords[col]
                m = np_argmax1(c).astype(np.float64)
                choices = np.where((m == 0) & (c[:, 0] == -1), typ-1, m)
            else:
                choices = coords[col].flatten()

            # Pick from the choices and try to have all of them atleast once
            if choices.size >= n:
                r = np.random.choice(choices, n, replace=False)
            else:
                n_replicates = n // choices.size
                r = np.random.permutation(np.concatenate((np.repeat(choices, n_replicates), np.random.choice(choices, n - choices.size * n_replicates))))
        
        # Fill design
        for i in range(n):
            Y[i*size: (i+1)*size, col] = r[i]

    return Y

@numba.njit(cache=CACHE)
def initialize_single(plot_sizes, factors, Y=None, coords=None):
    """
    Generate a random initial design for multiple-split plot problem.

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    plot_sizes : np.array
        An array specifying the number of plots at level
        i inside of one plot at level i+1. E.g a split-plot 
        design with 2 whole plots and 3 subplots is an array
        [3, 2].
    factors : np.array(2d)
        A 2D array with each row referring to the design factor
        column. Each factor specifies its level as a first element
        and the amount of levels as a second. 0 indicates a
        continuous factor.
    Y : np.array    
        An existing design if possible, otherwise it is generated
        as an all zeros-matrix.

    Returns
    -------
    Y : np.array
        A random design according to the multiple-split-plot.
    """
    ##################################################
    # INITIALIZATION
    ##################################################
    if Y is None:
        # Compute design sizes
        n = np.prod(plot_sizes)
        ncol = factors.shape[0]

        # Initiate design matrix
        Y = np.zeros((n, ncol), dtype=np.float64)

    # Compute alphas and betas
    alphas = np.cumprod(plot_sizes[::-1])[::-1]
    betas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))

    ##################################################
    # LOW-LEVEL FUNCTION
    ##################################################
    Y = __init_unconstrained(factors, Y, alphas, betas, coords=coords)

    return Y

@numba.njit(cache=CACHE)
def initialize(plot_sizes, factors, Y=None, n=1, coords=None):
    """
    Initialize multiple designs at the same time.

    .. note::
        This function is Numba accelerated
    """
    # Initialize larger design
    if Y is None:
        Y = np.zeros((n, np.prod(plot_sizes), factors.shape[0]), dtype=np.float64)

    # Loop and initialize
    for i in range(n):
        Y[i] = initialize_single(plot_sizes, factors, Y=Y[i], coords=coords)

    return Y

