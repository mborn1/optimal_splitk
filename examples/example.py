import numpy as np
import pandas as pd

from doe_gensplit.doe import *
from doe_gensplit.init import *
from doe_gensplit.utils import *
from doe_gensplit.validation import *

root = '.'

# The model
model = pd.read_excel(f'{root}/assets/model.xlsx', index_col=0).to_numpy()
columns = ['D', 'C', 'B', 'A', 'LOC', 'Cap', 'RPM']

# The factor columns
factors = np.array([
    [3, 1],         # D:   level 3; continuous
    [2, 1],         # C:   level 2; continuous
    [1, 1],         # B:   level 1; continuous
    [1, 1],         # A:   level 1; continuous
    [0, 1],         # LOC: level 1; continuous
    [0, 1],         # Capacity: level 1; continuous
    [0, 1],         # RPM: level 1; continuous
])

# Plot sizes
plot_sizes = np.array([3, 3, 3, 2], dtype=np.int64)

# Compute V-matrix (observation correlation matrix)
V = obs_var(plot_sizes)

# Validate the model (assertion)
validate_model(model, factors)

# Create optimal design
best_Y, metrics = doe(model, plot_sizes, factors, n_tries=1000)

# Validation
best_X = x2fx(encode_design(best_Y, factors), encode_model(model, factors))
det_val = np.linalg.det(best_X.T @ np.linalg.solve(V, best_X))

# Validation and output
print()
print('===============================')
print('Validation')
validate_design(best_Y, model, factors, plot_sizes)
print('-------------------------------')
print('Determinant:', np.max(metrics)**(1/len(model)), det_val**(1/len(model)))

# Output design
pd.DataFrame(best_Y, columns=columns).to_csv(f'{root}/out/example.csv', index=False)