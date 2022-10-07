import numpy as np
import pandas as pd
from doe_gensplit.utils import obs_var
from doe_gensplit.encode import encode_model, encode_design
from doe_gensplit.validation import validate_model
from doe_gensplit.doe import x2fx, doe

import argparse
parser = argparse.ArgumentParser(description="Perform a sensitivity analysis")
parser.add_argument('--start', type=float, help='The start fraction. Used for multiprocessing', default=0.0)
parser.add_argument('--stop', type=float, help='The stop fraction. Used for multiprocessing', default=1.0)
parser.add_argument('--ratios', type=float, nargs='+', help='The ratios to check for each level', default=[0.1, 1, 10])
parser.add_argument('--out', type=str, default='r.csv', help='The output file for the results')
parser.add_argument('--n', default=1000, type=int, help='The amount of random designs per ratio')
args = parser.parse_args()

#####################################################################
def create_max_model(labels_quad, labels_int, labels_lin):
    # Lengths
    nquad, ntwo, nlin = len(labels_quad), len(labels_int), len(labels_lin)

    # Compute terms
    nint = nquad + ntwo
    nmain = nlin + nint
    nterms = nmain + int(nint * (nint - 1) / 2) + nquad + 1

    # Initialize (pre-allocation)
    max_model = np.zeros((nterms, nmain), dtype=np.int64)
    stop = 1

    # Main effects
    max_model[stop + np.arange(nmain), np.arange(nmain)] = 1
    stop += nmain

    # Interaction effects
    for i in range(nint - 1):
        max_model[stop + np.arange(nint - 1 - i), i] = 1
        max_model[stop + np.arange(nint - 1 - i), i + 1 + np.arange(nint - 1 - i)] = 1
        stop += (nint - 1 - i)

    # Quadratic effects
    max_model[stop + np.arange(nquad), np.arange(nquad)] = 2

    return pd.DataFrame(max_model, columns=labels_quad + labels_int + labels_lin)


#####################################################################
labels_quad = ['A', 'B', 'C', 'D', 'E']
labels_int = []
labels_lin = []
model = create_max_model(labels_quad, labels_int, labels_lin)
columns = model.columns
model = model.to_numpy()

# The factor columns
factors = np.array([
    [3, 1],         # Drum
    [2, 1],         # Sharpness
    [1, 1],         # A
    [0, 1],         # LOC
    [0, 1],         # Capacity
])

# Plot sizes
plot_sizes = np.array([2, 2, 2, 4], dtype=np.int64)

# Compute V-matrix (observation correlation matrix)
V = obs_var(plot_sizes)

# Encode the model
model_enc = encode_model(model, factors)

# Validate the model
validate_model(model, factors)

# The ratios to validate
ratios = np.array(args.ratios)

#####################################################################

# Create a combination of all ratios to test
r = ratios[np.newaxis, :]
for i in range(len(plot_sizes) - 2):
    r = np.concatenate((
        np.tile(ratios, r.shape[1])[np.newaxis, :], 
        np.repeat(r, ratios.size).reshape(r.shape[0], -1)
    ))

# Partition for use in multiprocessing
r = r[:, int(args.start * r.shape[1]): int(args.stop * r.shape[1])]

# Loop over all combinations of true ratios
res = np.zeros((r.shape[1]**2, 2*r.shape[0] + 2))
i = 0
for true_ratio in r.T:
    # Compute the true observation matrix
    true_ratio = np.concatenate(([1], true_ratio))
    V_true = obs_var(plot_sizes, ratios=true_ratio)

    # Loop over all combinations of design ratios
    for design_ratio in r.T:
        # Compute the design
        design_ratio = np.concatenate(([1], design_ratio))
        best_Y, metrics = doe(model, plot_sizes, factors, n_tries=args.n, ratios=design_ratio)
        det_val = np.max(metrics) ** (1/len(model))

        # Compute the metric ratio
        best_X = x2fx(encode_design(best_Y, factors), model_enc)
        det_val_true = np.linalg.det(best_X.T @ np.linalg.solve(V_true, best_X)) ** (1/len(model))

        # Add to the results
        res[i, :r.shape[0]] = true_ratio[1:]
        res[i, r.shape[0]:2*r.shape[0]] = design_ratio[1:]
        res[i, 2*r.shape[0]:] = (det_val, det_val_true) 
        i += 1

    np.savetxt(args.out, res)