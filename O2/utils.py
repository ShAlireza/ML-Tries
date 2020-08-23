import os

import numpy as np
import pandas as pd


def prepare_data(path='../iris.data', download=False, head=0):
    if download:
        path = os.path.join('https://archive.ics.uci.edu', 'ml',
                            'machine-learning-databases', 'iris',
                            'iris.data')

    df = pd.read_csv(path, header=None, encoding='utf-8')

    if head:
        assert type(head) == int, "Head must be an integer"
        print(df.head(head))

    Y = df.iloc[:100, 4].values
    Y = np.where(Y == 'Iris-setosa', -1, 1)

    X = df.iloc[:100, [0, 2]].values

    return X, Y
