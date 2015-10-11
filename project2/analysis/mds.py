
import pickle

import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import MDS


def mult_scl(X, geo_names, land_use_names, labels):
    print('labels:')
    for i, label in zip(range(1, len(labels) + 1), labels):
        print('{}: {}'.format(i, label))
    mds = MDS()
    points = mds.fit(np.nan_to_num(X.T[list(geo_names.keys()) + list(land_use_names.keys())].T)).embedding_
    plt.scatter(points[:, 0], points[:, 1], s=20, c='g')
    for label, x, y in zip(range(1, len(labels) + 1), points[:, 0], points[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(10, 10),
            textcoords='offset points', ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            size='xx-small',
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.show()


def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        feature_names, categorical_names, numeric_names, geo_names, land_use_names, labels, X = pickle.load(pickle_sr)
        mult_scl(X, geo_names, land_use_names, labels)
        print()

if __name__ == '__main__':
    main()
