import numpy as np


def prepare_data(features=None, print_classes=False,
                 standardize=False, split=False):
    from sklearn import datasets
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    if not features:
        features = [2, 3]

    iris = datasets.load_iris()
    X_train = iris.data[:, features]
    Y_train = iris.target

    X_test, Y_test = None, None

    if print_classes:
        print('Class labels:', np.unique(Y_train))

    if split:
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_train, Y_train, train_size=0.3, random_state=1, stratify=Y_train
        )
    if standardize:
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test) if split else X_test

    if split:
        return X_train, X_test, Y_train, Y_test

    return X_train, Y_train
