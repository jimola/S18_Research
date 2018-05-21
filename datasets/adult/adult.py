import pandas as pd
import numpy as np
import random

column_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

original = pd.read_csv('adult.data', names = column_names, sep = r'\s*,\s*', engine = 'python', na_values = '?')
original_test = pd.read_csv('adult.test', names = column_names, sep = r'\s*,\s*', engine = 'python', na_values = '?')

original = pd.concat([original, original_test])

del original['fnlwgt']
del original["Education"]

binary = pd.get_dummies(original)

# Let's fix the Target as it will be converted to dummy vars too
binary["Target"] = binary["Target_>50K"]
del binary["Target_<=50K"]
del binary["Target_>50K"]

labels = binary["Target"]
binary = binary[binary.columns.difference(["Target"])]
