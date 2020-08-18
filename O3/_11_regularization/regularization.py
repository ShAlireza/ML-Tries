import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from O3 import prepare_data

X_train, X_test, Y_train, Y_test = prepare_data(standardize=True,
                                                split=True)

weights, params = [], []

for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10. ** c, random_state=1,
                            solver='lbfgs', multi_class='ovr')

    lr.fit(X_train, Y_train)

    weights.append(lr.coef_[1])
    params.append(10. ** c)

weights = np.array(weights)
params = np.array(params)

plt.plot(params, weights[:, 0],
         label='petal length')

plt.plot(params, weights[:, 1],
         linestyle='--', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
