import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt


def plot_decision_regions(X: np.ndarray, Y: np.ndarray, classifier,
                          test_idx=None,
                          resolution=0.02):
    from matplotlib.colors import ListedColormap

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    color_map = ListedColormap(colors[:len(np.unique(Y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step=resolution),
                           np.arange(x2_min, x2_max, step=resolution))

    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=color_map)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for i, cls in enumerate(np.unique(Y)):
        plt.scatter(x=X[Y == cls, 0], y=X[Y == cls, 1],
                    alpha=0.8, c=colors[i], marker=markers[i],
                    label=cls, edgecolors='black')

    # highlight test examples
    if test_idx:
        # plot all examples
        X_test, Y_test = X[test_idx, :], Y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')


def standardize(X: np.ndarray):
    X_std = X.copy()
    return (X_std - X_std.mean(axis=0)) / X_std.std(axis=0)


def identity(x):
    return x


def sigmoid(x, clip=None):
    if clip:
        assert type(clip) == list or type(
            clip) == tuple, "Boundary should be a list of tuple"
        assert len(clip) == 2, "Boundary list size must be 2"
        assert clip[0] < clip[1], "Invalid interval"

        return 1 / (1 + np.exp(-np.clip(x, clip[0], clip[1])))

    return 1 / (1 + np.exp(-x))
