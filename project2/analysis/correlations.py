
import pickle

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import pairwise_distances


def remove_nan_rows(x, y):
    f = ~(np.isnan(x) | np.isnan(y))
    return x[f], y[f]


def top_correlations(X, names, dist):
    dists = pairwise_distances(
        np.nan_to_num(X.T[list(names.keys())]),
        metric=lambda x, y: dist(*remove_nan_rows(x, y))
    )
    dists[np.triu_indices_from(dists)] = -np.inf
    dists_flattened = dists.flatten()
    indices = np.argsort(-dists_flattened)
    for i in indices[:10]:
        row, col = i//len(names), i % len(names)
        print('{: <80}{: <80}{}'.format(
            list(names.values())[col], list(names.values())[row], dists_flattened[i]
        ))


def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        feature_names, categorical_names, numeric_names, X = pickle.load(pickle_sr)
        print('categorical:')
        top_correlations(X, categorical_names, matthews_corrcoef)
        print()
        print('numeric:')
        top_correlations(X, numeric_names, lambda *args: pearsonr(*args)[0])
        print()

if __name__ == '__main__':
    main()
