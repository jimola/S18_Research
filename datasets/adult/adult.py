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

original_train = pd.read_csv(train_path,
                             names = column_names,
                             sep = r'\s*,\s*',
                             engine = 'python',
                             na_values = '?',
                             true_values = ['>50K'],
                             false_values = ['<=50K'])
original_test = pd.read_csv(test_path,
                            names = column_names,
                            comment = '|',
                            sep = r'\s*,\s*',
                            engine = 'python',
                            na_values = '?',
                            true_values = ['>50K.'],
                            false_values = ['<=50K.'])

original = pd.concat([original_train, original_test], ignore_index=True)

del original['fnlwgt']
del original["Education"]

binary = pd.get_dummies(original)
labels = binary["Target"]
binary = binary[binary.columns.difference(["Target"])]
