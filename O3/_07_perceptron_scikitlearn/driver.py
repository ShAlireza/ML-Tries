import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

from utils import plot_decision_regions

from O3 import prepare_data

X, Y = prepare_data(print_classes=True)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=1, stratify=Y
)

print('Labels count in y:', np.bincount(Y))
print('Labels count in y_train:', np.bincount(Y_train))
print('Labels count in y_test:', np.bincount(Y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, Y_train)

Y_pred = ppn.predict(X_test_std)
print(f'Mis classified examples: {(Y_test != Y_pred).sum()}')

print(f'Accuracy: {accuracy_score(Y_test, Y_pred)}')

print(f'Accuracy: {ppn.score(X_test_std, Y_test)}')

X_combined_std = np.vstack((X_train_std, X_test_std))
Y_combined_std = np.hstack((Y_train, Y_test))

plot_decision_regions(X=X_combined_std, Y=Y_combined_std,
                      classifier=ppn, test_idx=range(105, 150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
