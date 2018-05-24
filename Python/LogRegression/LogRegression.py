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

class DPLogisticRegression:
    """Differentially private logistic regression

    Learn a differentially private logistic regression model using output
    perturbation [1].  The method works by first fitting a traditional logistic
    regression model, and then adding Laplace noise of suitable variance to its
    coefficients.  (NB: The adjacency relation corresponding to this privacy
    guarantee is the one that allows records to be modified, but not removed.
    In particular, the size of the database is considered public information.)

    The current implementation has a few limitations:

    * It only supports binary classification.

    * It cannot fit an intercept coefficient.

    * It can only use l2 penalty.

    * It cannot assign different penalization weights to different features.

    * The model's input must be bounded in l2 norm (cf. the K parameter below).


    Parameters
    ----------

    epsilon : float
        The privacy budget.  Must be non negative.

    K : float, default 1.0
        Bound on the l2 norm of the input features; must be a positive number.
        The amount of added noise is proportional to K.  Note that the original
        analysis of the method assumed K = 1; the argument generalizes easily to
        this case.

    C : float, default 1.0
        Inverse of regularization strength; must be a positive number.
        The smaller C is, the stronger the regularization.  The amount of noise
        added is proportional to C.  (The original analysis of the added noise
        was formulated in terms of the inverse of C.)


    [1] K. Chaudhuri, C. Monteleoni, A. Sarwate.  Differentially Private
    Empirical Risk Minimization.  In Journal of Machine Learning 12, 2011.

    """
    def __init__(self, epsilon, K = 1.0, C = 1.0):
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

    def _enforce_norm(self, X):
        """Ensure that X respects norm bounds

        Throw an error if any row of X has a norm bigger than K.

        NB: This method violates differential privacy, and should not be used
        with private data.

        """
        max_norm = np.sqrt(np.square(X).sum(axis = 1)).max()
        if (max_norm > self.K):
            raise ValueError("The l2 norm of the rows X of must be bounded by K = %f; "
                             "the maximum was %f" % (self.K, max_norm))

    def fit(self, X, y):
        if (len(X) == 0):
            raise ValueError("Must provide non-empty X")

        self._enforce_norm(X) # FIXME

        self.logit = self.logit.fit(X, y)

        noise = dp.laplacian_l2(self.epsilon, \
                                n = len(self.logit.coef_[0]), \
                                sensitivity = 2 * self.K * self.logit.C / len(X))

        self.logit.coef_[0] = self.logit.coef_[0] + noise

        return self

    def predict(self, X):
        """Compute model predictions

        Parameters
        ----------

        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------

        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        # FIXME In the case where the intercept term is zero, it might be fine
        # to use the model on non-private data without enforcing the norm
        # restriction.
        self._enforce_norm(X)
        return self.logit.predict(X)

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------

        X : array-like, shape = (n_samples, n_features)
            Test samples.

        y : array-like, shape = (n_samples) or (n_samples, n_outputs)
            True labels for X.

        Returns
        -------

        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        self._enforce_norm(X)
        return self.logit.score(X, y)

def test():
    X = pd.get_dummies(ttt.features)
    y = ttt.label
    K = np.sqrt(np.square(X).sum(axis = 1)).max()
    plogit = DPLogisticRegression(epsilon = 0.1, K = K, C = 1.0)
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
