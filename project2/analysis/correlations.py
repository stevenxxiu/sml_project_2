
import pickle

import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import pairwise_distances


# noinspection PyTypeChecker
def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        feature_names, categorical_names, numeric_names, X = pickle.load(pickle_sr)
        dists = pairwise_distances(np.nan_to_num(X.T[list(categorical_names.keys())]), metric=matthews_corrcoef)
        dists[np.triu_indices_from(dists)] = -np.inf
        dists_flattened = dists.flatten()
        indices = np.argsort(-dists_flattened)
        for i in indices[:10]:
            row, col = i//len(categorical_names), i % len(categorical_names)
            print('{: <80}{: <80}{}'.format(
                list(categorical_names.values())[col], list(categorical_names.values())[row], dists_flattened[i]
            ))

if __name__ == '__main__':
    main()
