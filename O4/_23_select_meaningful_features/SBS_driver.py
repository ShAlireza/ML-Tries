import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from O4 import prepare_data, SBS

X_train, X_test, y_train, y_test = prepare_data(standardize=True,
                                                split=True)

knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(estimator=knn, k_features=1)

sbs.fit(X_train, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()
