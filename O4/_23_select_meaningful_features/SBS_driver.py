import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from O4 import prepare_data, SBS

X_train, X_test, y_train, y_test, df_wine = prepare_data(standardize=True,
                                                         split=True,
                                                         random_state=0,
                                                         dataframe=True)

knn = KNeighborsClassifier(n_neighbors=5)

sbs = SBS(estimator=knn, k_features=1)

sbs.fit(X_train, y_train)

k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.5, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.show()

k3 = list(sbs.subsets_[10])
print(df_wine.columns[1:][k3])

knn.fit(X_train, y_train)
print(f"Training accuracy: {knn.score(X_train, y_train)}")

print(f'Test accuracy: {knn.score(X_test, y_test)}')

knn.fit(X_train[:, k3], y_train)
print(f'Training accuracy: {knn.score(X_train[:, k3], y_train)}')

print(f'Test accuracy: {knn.score(X_test[:, k3], y_test)}')

# There are many more feature selection algorithms available
# via scikit-learn. Those include recursive backward elimination
# based on feature weights, tree-based methods to select features
# by importance, and univariate statistical tests.
# A good summary with illustrative examples can be found at
# http://scikit-learn.org/stable/modules/feature_selection.html.
# You can find implementations of several different flavors of sequential
# feature selection related to the simple SBS that we implemented
# previously in the Python package mlxtend at
# http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/
