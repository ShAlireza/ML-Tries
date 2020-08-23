import os

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


def prepare_data(path='../wine.data', download=False, head=0):
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

    return df