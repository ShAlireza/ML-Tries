import os

import numpy as np

import pandas as pd


def csv_file(string=None):
    from io import StringIO
    if not string:
        string = """A,B,C,D
        1.0,2.0,3.0,4.0
        5.0,6.0,,8.0
        10.0,11.0,12.0,
        ,,,"""

    return StringIO(string)


def categorical_dataframe():
    df = pd.DataFrame([
        ['green', 'M', 10.1, 'class2'],
        ['red', 'L', 13.5, 'class1'],
        ['blue', 'XL', 15.3, 'class2']])

    df.columns = ['color', 'size', 'price', 'label']
    return df


def prepare_data(path='../wine.data', download=False, head=0,
                 print_classes=False, standardize=False, split=False,
                 dataframe=False):
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    if download:
        path = os.path.join('https://archive.ics.uci.edu', 'ml',
                            'machine-learning-databases', 'wine',
                            'wine.data')

    df = pd.read_csv(path, header=None)

    df.columns = ['Class label', 'Alcohol',
                  'Malic acid', 'Ash',
                  'Alcalinity of ash', 'Magnesium',
                  'Total phenols', 'Flavanoids',
                  'Nonflavanoid phenols',
                  'Proanthocyanins',
                  'Color intensity', 'Hue',
                  'OD280/OD315 of diluted wines',
                  'Proline']

    if head:
        assert type(head) == int, "Head must be an integer"
        print(df.head(head))

    X_train = df.iloc[:, 1:].values
    y_train = df.iloc[:, 0].values

    X_test, y_test = None, None

    if print_classes:
        print('Class labels:', np.unique(y_train))

    if split:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, train_size=0.3, random_state=0, stratify=y_train
        )
    if standardize:
        sc = StandardScaler()
        sc.fit(X_train)
        X_train = sc.transform(X_train)
        X_test = sc.transform(X_test) if split else X_test

    if split:
        if dataframe:
            return X_train, X_test, y_train, y_test, df
        return X_train, X_test, y_train, y_test

    if dataframe:
        return X_train, y_train, df
    return X_train, y_train
