import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

from O4 import prepare_data

X_train, X_test, y_train, y_test = prepare_data(split=True)

# Normalization (Min-Max)

min_max_scaler = MinMaxScaler()

X_train_norm = min_max_scaler.fit_transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)

print(X_test[:5, :4], X_test_norm[:5, :4], sep='\n')

# Standardization

# Manual Standardization

example = np.array([0, 1, 2, 3, 4, 5])
print('standardized: ', (example - example.mean()) / example.std())
print('normalized: ',
      (example - np.min(example)) / (np.max(example) - np.min(example)))

# Using scikit-learn

standard_scaler = StandardScaler()

X_train_std = standard_scaler.fit_transform(X_train)
X_test_std = standard_scaler.transform(X_test)

# Again, it is also important to highlight that we fit the StandardScaler
# class only once—on the training data—and use those parameters to transform
# the test dataset or any new data point.


# Other, more advanced methods for feature scaling are available from scikit-
# learn, such as the RobustScaler . The RobustScaler is especially helpful and
# recommended if we are working with small datasets that contain many outliers.
# Similarly, if the machine learning algorithm applied to this dataset is prone
# to overfitting, the RobustScaler can be a good choice. Operating on each
# feature column independently, the RobustScaler removes the median value and
# scales the dataset according to the 1st and 3rd quartile of the dataset
# (that is, the 25th and 75th quartile, respectively) such that more extreme
# values and outliers become less pronounced. The interested reader can find
# more information about the RobustScaler in the official scikit-learn
# documentation at:
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html


robust_scaler = RobustScaler()

X_test_robust = robust_scaler.fit_transform(X_test)
X_train_robust = robust_scaler.fit(X_train)
