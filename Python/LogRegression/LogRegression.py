import sys
import os
sys.path.append(os.path.abspath("../../datasets/adult"))
import adult as adult

from sklearn import linear_model, model_selection
import pandas as pd
import numpy as np

original_data = adult.original
X = adult.binary.fillna(value = 0.0)
y = adult.labels.fillna(value = 0.0)
indices = range(0, len(X))

X_train, X_test, y_train, y_test, indices_train, indices_test = model_selection.train_test_split(X, y, indices, train_size=0.70)

logit1 = linear_model.LogisticRegression(C = 1.0)
logit1 = logit1.fit(X_train, y_train)

logit2 = linear_model.LogisticRegression(C = 1.5)
logit2 = logit2.fit(X_train, y_train)

logit3 = linear_model.LogisticRegression(C = 0.5)
logit3 = logit3.fit(X_train, y_train)

class LogRegressionChooser:
    def __init__(hyperparams, mfeatures, train_dbs, scorefunc):
        pass
    def choose(DB):
        pass

    def fitPrivateModel(DB, epsilon):
        #Let's make this first
        pass
