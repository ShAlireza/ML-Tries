import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import plot_decision_regions

from O2._01_perceptron.perceptron import Perceptron
from O2.utils import prepare_data

X, Y = prepare_data()

plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')

plt.scatter(X[50:, 0], X[50:, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, epochs=10)
ppn.fit(X, Y)
plt.plot(range(1, len(ppn.errors_) + 1),
         ppn.errors_, marker='o')

plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

plot_decision_regions(X, Y, classifier=ppn)
plt.show()
