#!/usr/bin/python
import numpy as np
import pandas as pd
from optimal_splitk.utils import obs_var
from optimal_splitk.encode import encode_model, encode_design
from optimal_splitk.validation import validate_model
from optimal_splitk.doe import x2fx, doe

import argparse
parser = argparse.ArgumentParser(description="Perform a sensitivity analysis")
parser.add_argument('--proc', type=int, help='The current process', default=0)
parser.add_argument('--total', type=int, help='The total amount of processes', default=1)
parser.add_argument('--ratios', type=float, nargs='+', help='The ratios to check for each level', default=[0.1, 1, 10])
parser.add_argument('--root', type=str, default='.', help='The output file for the results')
parser.add_argument('--n', default=1000, type=int, help='The amount of random designs per ratio')
parser.add_argument('--reps', default=1, type=int, help='The number of repetitions for the entire experiment')
args = parser.parse_args()

print(args)

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
    [2, 1],
    [1, 1],
    [0, 1], 
    [0, 1], 
    [0, 1],
])

# Plot sizes
plot_sizes = np.array([2, 2, 6], dtype=np.int64)

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
total_size = r.shape[1] * args.reps
start = int(args.proc / args.total * total_size)
stop = int((args.proc + 1) / args.total * total_size)
print('Range:', start, stop)

# The output file
outfile = f'{args.root}/r{args.proc}.csv'

# Compute true covariances
V_trues = [(true_ratio, obs_var(plot_sizes, ratios=np.concatenate(([1], true_ratio)))) for true_ratio in r.T]

# Loop over all combinations of true ratios
res = np.zeros(((stop - start) * r.shape[1], 2*r.shape[0] + 2))
i = 0
j = 0
for rep in range(args.reps):
    # Loop over all combinations of design ratios
    for design_ratio in r.T:
        if start <= i < stop:
            # Compute the design
            design_ratio = np.concatenate(([1], design_ratio))
            best_Y, metrics = doe(model, plot_sizes, factors, n_tries=args.n, ratios=design_ratio)
            det_val = np.max(metrics) ** (1/len(model))

            # Compute the metric ratio
            best_X = x2fx(encode_design(best_Y, factors), model_enc)

            for true_ratio, V_true in V_trues:
                # Compute True D-criterion
                det_val_true = np.linalg.det(best_X.T @ np.linalg.solve(V_true, best_X)) ** (1/len(model))

                # Add to the results
                res[j, :r.shape[0]] = design_ratio[1:]
                res[j, r.shape[0]:2*r.shape[0]] = true_ratio
                res[j, 2*r.shape[0]:] = (det_val, det_val_true)
                j += 1
        i += 1

    np.savetxt(outfile, res)
