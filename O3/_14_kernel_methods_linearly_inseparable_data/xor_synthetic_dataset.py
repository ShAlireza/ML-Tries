import numpy as np

import matplotlib.pyplot as plt

from O3 import xor_dataset

np.random.seed(1)

X_xor, y_xor = xor_dataset()

plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='blue', marker='x', label='1')

plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='red', marker='s', label='-1')

plt.xlim([-3, 3])
plt.ylim([-3, 3])

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
