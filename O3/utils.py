import numpy as np


def prepare_data(features=None, print_classes=False):
    from sklearn import datasets

    if not features:
        features = [2, 3]

    iris = datasets.load_iris()
    X = iris.data[:, features]
    Y = iris.target
    if print_classes:
        print('Class labels:', np.unique(Y))

    return X, Y
