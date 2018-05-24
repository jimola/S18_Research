import sys
import os
sys.path.append(os.path.abspath("../../datasets/adult"))
import adult as adult
sys.path.append(os.path.abspath("../"))
import DPrivacy as dp
import LoadData as data

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pandas as pd
import numpy as np

class DataSet:
    def __init__(self, df, y_col = None):
        if (y_col == None):
            y_col = df.columns[-1]
        self.features = df[df.columns.difference([y_col])]
        self.label    = df[y_col]

data_path = os.path.abspath("../../datasets")

adult = DataSet(adult.original, y_col = "Target")
ttt = pd.read_csv(os.path.join(data_path, "tic-tac-toe.data"), header = None)
ttt = DataSet(ttt)
nurs = DataSet(pd.read_csv(os.path.join(data_path, "nursery.data"), header = None))
loan = DataSet(pd.read_csv(os.path.join(data_path, "student-loan.csv")))

def tuning(d):
    dummies = pd.get_dummies(d.features)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dummies, d.label)
    Cs = [0.5, 1.0, 1.5, 2.0]
    models = [LogisticRegression(C = C).fit(X_train, y_train) for C in Cs]
    return X_test, y_test, Cs, models

# TODO
#
# - Ensure that we are doing binary classification
# - Be clear on Hamming vs. non-Hamming distance
class DPLogisticRegression:
    def __init__(self, epsilon, K, C = 1.0):
        self.logit = LogisticRegression(penalty = 'l2',
                                        C = C,
                                        dual = False,
                                        tol = 1e-4, # XXX Does this have an impact on sensitivity?
                                        fit_intercept = False,
                                        class_weight = None,
                                        solver = 'liblinear')
        if (epsilon < 0):
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon
        if (K <= 0):
            raise ValueError("K must be positive")
        self.K = K

    def fit(self, X, y):
        if (len(X) == 0):
            raise ValueError("Must provide non-empty X")
        max_norm = np.sqrt(np.square(X).sum(axis = 1)).max()
        if (max_norm > self.K):
            raise ValueError("The l2 norm of the rows X of must be bounded by K = %f; "
                             "the maximum was %f" % (self.K, max_norm))

        self.logit = self.logit.fit(X, y)

        noise = dp.laplacian_l2(self.epsilon, \
                                n = len(self.logit.coef_[0]), \
                                sensitivity = 2 * self.K * self.logit.C / len(X))

        print self.logit.coef_[0], noise

        self.logit.coef_[0] = self.logit.coef_[0] + noise

        return self

    def predict(self, X):
        return self.logit.predict(X)

    def score(self, X, y):
        return self.logit.score(X, y)

def test():
    X = pd.get_dummies(ttt.features)
    y = ttt.label
    K = np.sqrt(np.square(X).sum(axis = 1)).max()
    plogit = DPLogisticRegression(epsilon = 200, K = K, C = 1.0)
    plogit = plogit.fit(X, y)
    return X, y, plogit

class LogRegressionChooser:
    def __init__(hyperparams, mfeatures, train_dbs, scorefunc):
        pass
    def choose(DB):
        pass

    def fitPrivateModel(DB, epsilon):
        #Let's make this first
        pass
