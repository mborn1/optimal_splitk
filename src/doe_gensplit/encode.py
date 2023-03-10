import numpy as np
import numba
from .utils import np_take_advanced, CACHE


def encode_model(model, factors):
    """
    Encode the model-matrix according to the factors.
    Each continuous variable is encoded as a single columns,
    each categorical variable is encoded using effect-encoding,
    creating n-1 columns (with n the amount of categorical levels).

    Parameters
    ----------
    model : np.array(2d)
        The initial model, before encoding
    factors : np.array(2d)
        An array of factors indicating whether they are
        continuous or categorical, and the amount of
        categorical levels.

    Returns
    -------
    model : np.array
        The newly encoded model.
    """
    # Extract factor type
    options = factors[:, 1]
    # Number of columns required for encoding
    cols = np.where(options > 1, options - 1, options)

    ####################################

    # Insert extra columns for the encoding
    extra_columns = cols - 1
    a = np.zeros(np.sum(extra_columns), dtype=np.int64)
    start = 0
    for i in range(extra_columns.size):
        a[start:start+extra_columns[i]] = np.full(extra_columns[i], i+1)
        start += extra_columns[i]
    model = np.insert(model, a, 0, axis=1)

    ####################################

    # Loop over all terms and insert identity matrix (with rows)
    # if the term is present
    current_col = 0
    # Loop over all factors
    for i in range(cols.size):
        # If required more than one column
        if cols[i] > 1:
            j = 0
            # Loop over all rows
            while j < model.shape[0]:
                if model[j, current_col] == 1:
                    # Replace ones by identity matrices
                    ncols = cols[i]
                    model = np.insert(model, [j] * (ncols - 1), model[j], axis=0)
                    model[j:j+ncols, current_col:current_col+ncols] = np.eye(ncols)
                    j += ncols
                else:
                    j += 1
            current_col += cols[i]
        else:
            current_col += 1

    return model

@numba.njit(cache=CACHE)
def encode_design(Y, factors):
    """
    Encode the design according to the factors.
    Each categorical factor is encoded using
    effect-encoding. The expected input is an
    integer describing the chosen category.

    .. note::
        This function is Numba accelerated

    Parameters
    ----------
    Y : np.array
        The current design matrix
    factors : np.array 
        An array of factors indicating whether they are
        continuous or categorical, and the amount of
        categorical levels.

    Returns
    -------
    Yenc : np.array
        The encoded design-matrix 
    """
    # Compute amount of columns per factor
    options = factors[:, 1]
    cols = np.where(options > 1, options - 1, options)

    # Initialize encoding
    ncols = np.sum(cols)
    Yenc = np.zeros((*Y.shape[:-1], ncols))

    start = 0
    # Loop over factors
    for i in range(options.size):
        if options[i] == 1:
            # Continuous factor: copy
            Yenc[..., start] = Y[..., i]
            start += 1
        else:
            # Categorical factor: effect encode
            eye = np.concatenate((np.eye(cols[i]), -np.ones((1, cols[i]))))
            Yenc[..., start:start+cols[i]] = np_take_advanced(eye, Y[..., i].astype(np.int64))
            start += cols[i]

    return Yenc

@numba.njit(cache=CACHE)
def decode_design(Y, factors):
    """
    Decode the design according to the factors.
    Each categorical factor is decoded from
    effect-encoding. The expected input is an
    effect-encoded design matrix.

    It is the inverse of :py:func:`encode_design`

    .. note::
        This function in Numba accelerated

    Parameters
    ----------
    Y : np.array
        The current, effect-encoded design matrix.
    factors : np.array 
        An array of factors indicating whether they are
        continuous or categorical, and the amount of
        categorical levels.

    Returns
    -------
    Ydec : np.array
        The decoded design-matrix 
    """
    # Compute amount of columns per factor
    options = factors[:, 1]

    # Initialize dencoding
    Ydec = np.zeros((*Y.shape[:-1], factors.shape[0]))

    # Loop over all factors
    start = 0
    for i in range(options.size):
        if options[i] == 1:
            Ydec[..., i] = Y[..., start]
            start += 1
        else:
            ncols = options[i] - 1
            Ydec[..., i] = np.where(Y[..., start] == -1, ncols, np.argmax(Y[..., start:start+ncols], axis=-1))
            start += ncols

    return Ydec