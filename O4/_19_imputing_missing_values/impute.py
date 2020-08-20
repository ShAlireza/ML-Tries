import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer

from O4 import csv_file

csv = csv_file()

df = pd.read_csv(csv)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer = imputer.fit(df.values)

imputed_data = imputer.transform(df.values)

print(imputed_data)
