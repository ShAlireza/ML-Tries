import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

from O4 import categorical_dataframe

df = categorical_dataframe()

print(df)

# Mapping for ordinal features should be done by our selves

size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}

df['size'] = df['size'].map(size_mapping)

print(df)

# If we want to transform the integers back to string we can
# use reverse mapping like below example:

reverse_size_mapping = {v: k for k, v in size_mapping.items()}

back_df = df.copy()
back_df['size'] = back_df.get('size').map(reverse_size_mapping)
print(back_df)

# Many machine learning libraries require that class labels are encoded as
# integer values. Although most estimators for classification in scikit-learn
# convert class labels to integers internally, it is considered good practice
# to provide class labels as integer arrays to avoid technical glitches. To
# encode the class labels, we can use an approach similar to the mapping of
# ordinal features.

# We need to remember that class labels are not ordinal, and it doesn't matter
# which integer number we assign to a particular string label. Thus, we can
# simply enumerate the class labels, starting at 0 :

class_mapping = {label: idx for idx, label in
                 enumerate(np.unique(df['label']))}

print(class_mapping)

manual_df = df.copy()

manual_df['label'] = manual_df['label'].map(class_mapping)

print(manual_df)

# And we can simply invert integers to string like above example
reverse_class_mapping = {v: k for k, v in class_mapping.items()}

back_df = manual_df.copy()
back_df['label'] = back_df.get('label').map(reverse_class_mapping)
print(back_df)

# Or simply using scikit-learn LabelEncoder

class_label_encoder = LabelEncoder()

df['label'] = class_label_encoder.fit_transform(df['label'].values)

print(df)

# And even inverse of it

y = class_label_encoder.inverse_transform(df['label'])

print(y)

# In the previous Mapping ordinal features section, we used a simple
# dictionary-mapping approach to convert the ordinal size feature into
# integers. Since scikit-learn's estimators for classification treat class
# labels as categorical data that does not imply any order (nominal), we used
# the convenient LabelEncoder to encode the string labels into integers. It may
# appear that we could use a similar approach to transform the nominal color
# column of our dataset, as follows:

X = df[['color', 'size', 'price']].values
color_label_encoder = LabelEncoder()
X[:, 0] = color_label_encoder.fit_transform(X[:, 0])

print(X)

# Above example is wrong, because there is no natural ordering between
# any of colors, but here we are assuming this order: red > green > blue !
# Instead we are going to use one-hot encoding technique.

X = df[['color', 'size', 'price']].values

color_one_hot_encoder = OneHotEncoder()
print(X[:, 0].reshape(-1, 1))
one_hot = color_one_hot_encoder.fit_transform(X[:, 0].reshape(-1, 1))

print(one_hot.toarray())  # Returned value is a sparse array
# so we turned it to array

# If we want to selectively transform columns in a multi-feature array,
# we can use the ColumnTransformer , which accepts a list of
# (name, transformer or ('drop', 'passthrough'), column(s)) tuples as follows:

X = df[['color', 'size', 'price']].values

column_transformer = ColumnTransformer(transformers=[
    ('onehot', OneHotEncoder(), (0,)),
    ('nothing', 'passthrough', (1, 2))
], remainder='passthrough')  # Remainder estimator will apply on unspecified
# columns

transformed = column_transformer.fit_transform(X)
print(transformed)

# An even more convenient way to create those dummy features via
# one-hot encoding is to use the get_dummies method implemented in pandas.
# Applied to a DataFrame ,the get_dummies method will only convert string
# columns and leave all other columns unchanged.

print(pd.get_dummies(df[['price', 'size', 'color']]))

# When we are using one-hot encoding datasets, we have to keep in mind that
# this introduces multicollinearity, which can be an issue for certain methods
# (for instance, methods that require matrix inversion). If features are highly
# correlated, matrices are computationally difficult to invert, which can lead
# to numerically unstable estimates. To reduce the correlation among variables,
# we can simply remove one feature column from the one-hot encoded array. Note
# that we do not lose any important information by removing a feature column,
# though; for example, if we remove the column color_blue , the feature
# information is still preserved since if we observe color_green=0 and color_
# red=0 , it implies that the observation must be blue.

# If we use the get_dummies function, we can drop the first column by passing
# a True argument to the drop_first parameter, as shown in the following code
# example:

print(pd.get_dummies(df[['price', 'size', 'color']], drop_first=True))

# In order to drop a redundant column via the OneHotEncoder , we need to set
# drop='first' and set categories='auto' as follows:

X = df[['color', 'size', 'price']].values

color_one_hot_encoder = OneHotEncoder(categories='auto', drop='first')
column_transformer = ColumnTransformer(transformers=[
    ('onehot', color_one_hot_encoder, [0]),
    ('nothing', 'passthrough', [1, 2])
])

print(column_transformer.fit_transform(X))

# If we are unsure about the numerical differences between
# the categories of ordinal features, or the difference between
# two ordinal values is not defined, we can also encode them
# using a threshold encoding with 0/1 values. For example, we
# can split the feature size with values M, L, and XL into two
# new features, "x > M" and "x > L".
# We can use the apply method of pandas' DataFrames to write
# custom lambda expressions in order to encode these variables
# using the value-threshold approach:

df = categorical_dataframe()

df['x > M'] = df['size'].apply(
    lambda x: 1 if x in {'L', 'XL'} else 0)
df['x > L'] = df['size'].apply(
    lambda x: 1 if x == 'XL' else 0)
del df['size']
print(df)
