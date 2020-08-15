import numpy as np

import matplotlib.pyplot as plt

from O2.utils import prepare_data
from O2._04_adaline_SGD.adaline_SGD import AdalineSGD

from utils import plot_decision_regions, standardize

X, Y = prepare_data()

X_std = standardize(X)

adaline = AdalineSGD(eta=0.01, epochs=15, random_state=1)
adaline.fit(X_std, Y)

plot_decision_regions(X_std, Y, classifier=adaline)
plt.title('Adaline - Mini Batch Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend('upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(adaline.cost_) + 1), adaline.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()
