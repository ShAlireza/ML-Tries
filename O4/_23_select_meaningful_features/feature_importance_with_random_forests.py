"""

    Another useful approach for selecting relevant features from a dataset is
    using a random forest, an ensemble technique. Using a random forest, we can
    measure the feature importance as the averaged impurity decrease computed
    from all decision trees in the forest, without making any assumptions about
    whether our data is linearly separable or not. Conveniently, the random
    forest implementation in scikit-learn already collects the feature
    importance values for us so that we can access them via the
    feature_importances_ attribute after fitting a RandomForestClassifier .
    By executing the following code, we will now train a forest of 500 trees on
    the Wine dataset and rank the 13 features by their respective importance
    measures—remember. Also we know that we don't need to use standardized or
    normalized features in tree-based models

"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from O4 import prepare_data

X_train, X_test, y_train, y_test, df_wine = prepare_data(dataframe=True,
                                                         split=True)

feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices],
        align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices],
           rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()
