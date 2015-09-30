
import pickle

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


# noinspection PyTypeChecker
def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        feature_names, categorical_feats, X = pickle.load(pickle_sr)
        dists = pairwise_distances(np.nan_to_num(X).T, metric='cosine')
        dists[np.triu_indices_from(dists)] = -np.inf
        dists_flattened = dists.flatten()
        indices = np.argsort(-dists_flattened)
        for i in indices[:10]:
            row, col = i//len(feature_names), i % len(feature_names)
            print(feature_names[col], feature_names[row], dists_flattened[i])

if __name__ == '__main__':
    main()
