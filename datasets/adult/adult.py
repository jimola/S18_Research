import os
import pandas as pd
import numpy as np
import random

column_names = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per week", "Country", "Target"]

path = os.path.dirname(os.path.realpath(__file__))
train_path = os.path.join(path, 'adult.data')
test_path = os.path.join(path, 'adult.test')

original = pd.read_csv(train_path, names = column_names, sep = r'\s*,\s*', engine = 'python', na_values = '?')
original_test = pd.read_csv(test_path, names = column_names, sep = r'\s*,\s*', engine = 'python', na_values = '?')

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
