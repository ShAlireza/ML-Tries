from io import StringIO

import pandas as pd

csv_data = """A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,
,,,"""

df = pd.read_csv(StringIO(csv_data))

print(df)

# Print number of NaNs in each column

print(df.isnull().sum())

# Drop rows with NaN

print(df.dropna(axis=0))

# Drop columns with NaN

print(df.dropna(axis=1))

# only drop rows where all columns are NaN
# (Removes last row of data frame)

print(df.dropna(how='all'))

# Drop rows that have fewer than  k real values
k = 4
print(df.dropna(thresh=k))

# Only drop rows where NaN appear in specific columns (here: 'C')

print(df.dropna(subset=['C']))
