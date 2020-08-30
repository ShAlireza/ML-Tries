import numpy as np

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

from O4 import prepare_data

X_train, X_test, y_train, y_test, df_wine = prepare_data(split=True,
                                                         dataframe=True)

forest = RandomForestClassifier(n_estimators=500, random_state=1)
forest.fit(X_train, y_train)

feat_labels = df_wine.columns[1:]

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
print('Number of features that meet this threshold criterion:',
      X_selected.shape[1])

for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30,
                            feat_labels[indices[f]],
                            importances[indices[f]]))
