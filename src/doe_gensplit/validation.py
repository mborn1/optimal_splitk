import numpy as np


def validate_model(model, factors, encoded=False):
    """
    Validate the model.

    * Make sure each factor is displayed in the model
    * Make sure the categorical variables only have linear effects (max 1 in model)

    Parameters
    ----------
    model : np.array
        The regression model (MATLAB notation)
    factors : np.array
        An array of factors indicating whether they are
        continuous or categorical, and the amount of
        categorical levels.
    encoded : bool
        Flag indicating whether the model is encoded

    Raises
    ------
    AssertionError
    """
    # Start of each column
    if encoded:
        col_start = np.concatenate((np.array([0]), 
                                    np.cumsum(np.where(factors[:, 1] > 1, 
                                                       factors[:, 1] - 1, 
                                                       np.ones(factors.shape[0], dtype=np.int64)))))
    else:
        col_start = np.arange(factors.shape[0] + 1)

    # Check all terms present
    assert model.shape[1] == col_start[-1], f"Amount of terms in factors ({col_start[-1]}) != model ({model.shape[1]})"

    # Check that a categorical factor does not contain any quadratics
    for i in range(factors.shape[0]):
        if factors[i, 1] > 1:
            assert np.all(model[:, col_start[i]] <= 1), f"""Factor {i} cannot have quadratic effect"""

    # Check that categorical dummy variables all have the same value
    if encoded:
        for i in range(factors.shape[0]):
            if factors[i, 1] > 1:
                idx = np.where(model[:, col_start[i]] >= 1)[0]
                for j in range(1, factors[i, 1] - 1):
                    assert np.all(model[idx + j, col_start[i] + j] == model[idx, col_start[i]]), f'Dummy variables for factor {i} are badly encoded'

    return True

   
def validate_design(Y, model, factors, plot_sizes, encoded=False):
    """
    Validate the design.

    * Make sure the design is correctly grouped.

    Parameters
    ----------
    Y : np.array
        The design matrix to validate
    model : np.array
        The regression model (MATLAB notation)
    factors : np.array
        An array of factors indicating whether they are
        continuous or categorical, and the amount of
        categorical levels.
    plot_sizes : np.array
        The size of each level
    encoded : bool
        Whether the design is effect-encoded or not
    
    Raises
    ------
    AssertionError
    """
    # Start of each column
    if encoded:
        col_start = np.concatenate((np.array([0]), 
                                    np.cumsum(np.where(factors[:, 1] > 1, 
                                                       factors[:, 1] - 1, 
                                                       np.ones(factors.shape[0], dtype=np.int64)))))
    else:
        col_start = np.arange(factors.shape[0] + 1)

    # Betas
    betas = np.cumprod(np.concatenate((np.array([1]), plot_sizes)))

    # Make sure Y has enough rows
    assert Y.shape[0] == betas[-1], f'Runs do not match with plot sizes: Y ({Y.shape[0]}) - sizes ({betas[-1]})'

    # Check the columns of Y
    assert Y.shape[1] == col_start[-1], f'The design does not have enough effects: Y ({Y.shape[1]}) - ({col_start[-1]})'

    # Make sure the plots are grouped correctly
    for i in range(factors.shape[0]):
        level = factors[i, 0]
        b = betas[level]

        # Check that it is a multiple of the plot size
        assert np.mod(Y.shape[0] / b, 1) == 0, f'Shape of Y is not multiple of plot size at position {i} ({b})'

        x = Y[:, col_start[i]:col_start[i+1]].reshape((-1, b, col_start[i+1] - col_start[i]))
        x = np.moveaxis(x, 0, 1)
        assert np.all(np.logical_or(np.all(x == x[0], axis=(0, 2)), np.all(np.isnan(x), axis=(0, 2)))), f'Factor {i} is not correctly grouped'

    return True