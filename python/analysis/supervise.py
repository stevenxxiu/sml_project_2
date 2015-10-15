
import pickle

import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR


def main():
    with open('data/binarized.pickle', 'rb') as pickle_sr:
        labels, names, categorical_names, numeric_names, X = pickle.load(pickle_sr)
        X = np.nan_to_num(X)
    i = names.index('Unemployed, %')
    X, y = np.delete(X, i, 1), X[:, i]
    estimator = SVR(kernel='linear', C=1)
    selector = RFECV(estimator, step=1, cv=5, scoring='mean_squared_error')
    selector = selector.fit(X, y)
    print(selector.support_)


if __name__ == '__main__':
    main()
