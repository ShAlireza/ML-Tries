import numpy as np

import matplotlib.pyplot as plt

np.random.seed(1)

X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='blue', marker='x', label='1')

plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='red', marker='s', label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
