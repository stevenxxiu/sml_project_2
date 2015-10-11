
import pickle

import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from matplotlib import pyplot as plt

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return [x, y]


def plot_location(location, labels, ax):

    compass_points = [
        "E",
        "ENE",
        "NE",
        "NNE",
        "N",
        "NNW",
        "NW",
        "WNW",
        "W",
        "WSW",
        "SW",
        "SSW",
        "S",
        "SSE",
        "SE",
        "ESE",
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
    for label, x, y in zip(range(1,len(labels) + 1), points[:, 0], points[:, 1]):
        ax.annotate(
            label,
            xy = (x, y), xytext = (10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            size = 'xx-small',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

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
    for label, x, y in zip(range(1,len(labels) + 1), points2[:, 0], points2[:, 1]):
        ax1.annotate(
            label,
            xy = (x, y), xytext = (10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            size = 'xx-small',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    mds = MDS()
    points1 = mds.fit(np.nan_to_num(X.T[list(geo_names.keys()) + list(land_use_names.keys())].T)).embedding_
    ax2.scatter(points1[:, 0], points1[:, 1], s=20, c='g')
    ax2.set_title('MDS')
    for label, x, y in zip(range(1,len(labels) + 1), points1[:, 0], points1[:, 1]):
        ax2.annotate(
            label,
            xy = (x, y), xytext = (10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            size = 'xx-small',
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    plt.show()


def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        feature_names, categorical_names, numeric_names, geo_names, land_use_names, labels, location, X = pickle.load(pickle_sr)
        #plot_location(X.T["Location (distance km)"], X.T["Location (direction)"], labels)
        mult_scl(X, geo_names, land_use_names, labels, location)
        print()

if __name__ == '__main__':
    main()
