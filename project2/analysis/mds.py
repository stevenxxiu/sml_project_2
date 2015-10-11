
import pickle

import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]


def plot_location(location, labels, ax):

    compass_points = [
        "E","ENE","NE","NNE",
        "N","NNW","NW","WNW",
        "W","WSW","SW","SSW",
        "S","SSE","SE","ESE",
    ]

    points_degrees = {}
    degrees = 0.0
    for compass_point in compass_points:
        points_degrees[compass_point] = degrees
        degrees += np.pi / 8

    print(points_degrees)

    points = np.array([pol2cart(pt[0], points_degrees[pt[1]]) for pt in location])

    ax.scatter(points[:, 0], points[:, 1], s=20, c='b')
    ax.set_title('Location')
    add_labels(labels, points, ax)

def mult_scl(X, geo_names, land_use_names, labels, location):

    print('labels:')
    for i, label in zip(range(1, len(labels) + 1), labels):
        print "{}: {}".format(i, label)


    isomap = Isomap()
    points2 = isomap.fit(np.nan_to_num(X.T[list(geo_names.keys()) + list(land_use_names.keys())].T)).embedding_

    f, (ax1, ax2, ax3) = plt.subplots(1, 3)

    plot_location(location, labels, ax3)

    ax1.scatter(points2[:, 0], points2[:, 1], s=20, c='r')
    ax1.set_title('Isomap')
    add_labels(labels, points2, ax1)

    mds = MDS()
    points1 = mds.fit(np.nan_to_num(X.T[list(geo_names.keys()) + list(land_use_names.keys())].T)).embedding_
    ax2.scatter(points1[:, 0], points1[:, 1], s=20, c='g')
    ax2.set_title('MDS')
    add_labels(labels, points1, ax2)

    plt.show()

def add_labels(labels, points, ax):
    for label, x, y in zip(range(1,len(labels) + 1), points[:, 0], points[:, 1]):
        ax.annotate(
            label,
            xy = (x, y), xytext = (10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            size = 'xx-small',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

def normalise(X, numeric_names):
    num_keys = list(numeric_names.keys())
    i = 0
    Xp = []
    for row in X.T:
        if i in num_keys:
            #print (normalize(np.nan_to_num(row))[0])
            Xp.append(normalize(np.nan_to_num(row))[0])
        else:
            Xp.append(np.nan_to_num(0.2*row))
        i += 1
    Xp = np.array(Xp)
    return Xp.T

def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        feature_names, categorical_names, numeric_names, geo_names, land_use_names, labels, location, X = pickle.load(pickle_sr)
        print(X.shape)
        X = normalise(X, numeric_names)
        print(X.shape)
        mult_scl(X, geo_names, land_use_names, labels, location)
        print()

if __name__ == '__main__':
    main()
