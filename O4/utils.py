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
