import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from O2._02_adaline.adaline import AdalineGD

# path = os.path.join('https://archive.ics.uci.edu', 'ml',
#                     'machine-learning-databases', 'iris', 'iris.data')

path = '../iris.data'

df = pd.read_csv(path, header=None, encoding='utf-8')

print(df.head())

Y = df.iloc[:100, 4].values
Y = np.where(Y == 'Iris-setosa', -1, 1)

X = df.iloc[:100, [0, 2]].values

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ada1 = AdalineGD(eta=0.01, epochs=10).fit(X, Y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(SSE)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = AdalineGD(eta=0.0001, epochs=10).fit(X, Y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('SSE')
ax[1].set_title('Adaline - Learning rate 0.0001')

plt.show()
