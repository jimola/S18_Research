import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
import pandas as pd
import numpy as np
from data import adult, ttt, nursery, loan

import DPrivacy as dp

class DataSet:
    def __init__(self, df, y_col = None):
        if y_col == None:
            y_col = df.columns[-1]
        self.features = df[df.columns.difference([y_col])]
        self.label    = df[y_col]

adult = DataSet(adult.original, y_col = "Target")
ttt = DataSet(ttt.data)
nurs = DataSet(nursery.data)
loan = DataSet(loan.data)

def tuning(d):
    dummies = pd.get_dummies(d.features)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(dummies, d.label)
    Cs = [0.5, 1.0, 1.5, 2.0]
    models = [LogisticRegression(C = C).fit(X_train, y_train) for C in Cs]
    return X_test, y_test, Cs, models

class DPLogisticRegression:
    """Differentially private logistic regression

    Learn a differentially private logistic regression model using output
    perturbation.  The method works by first fitting a traditional logistic
    regression model, and then adding Laplace noise of suitable variance to its
    coefficients.  (NB: The adjacency relation corresponding to this privacy
    guarantee is the one that allows records to be modified, but not removed.
    In particular, the size of the database is considered public information.)

    The current implementation has a few limitations:

    * It only supports one-vs.-rest fitting for multi-class problems.

    * It can only use l2 penalty.

    * It cannot assign different penalization weights to different features.

    * The model's input must be bounded in l2 norm (cf. the K parameter below).

    There is a potential pitfall when fitting a multi-class model.  Since we
    currently fit one separate model for each class, we have to split the
    privacy budget among all of them, which leads to larger noise.  It seems
    that this could be reduced by using a multinominal cost function, since
    optimizing it has a sensitivity that is only twice the one of the binary
    case.

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


    fit_intercept : boolean, default False
        Whether to fit an intercept coefficient in the model.  Note that fitting
        an intercept requires adding slightly more noise to achieve the same privacy
        level: without an intercept, the noise is proporitional to K; with an
        intercept, it is proportional to sqrt(K^2 + 1)


    References
    ----------

    K. Chaudhuri, C. Monteleoni, A. Sarwate.  Differentially Private
    Empirical Risk Minimization.  In Journal of Machine Learning 12, 2011.

    """
    def __init__(self, epsilon, K = 1.0, C = 1.0, fit_intercept = False):
        self.logit = LogisticRegression(penalty = 'l2',
                                        C = C,
                                        dual = False,
                                        tol = 1e-4, # XXX Does this have an impact on sensitivity?
                                        fit_intercept = fit_intercept,
                                        class_weight = None,
                                        solver = 'liblinear')
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        self.epsilon = epsilon
        if K <= 0:
            raise ValueError("K must be positive")
        self.K = K
        if fit_intercept:
            self.K_eff = np.sqrt(K * K + 1)
        else:
            self.K_eff = K

    def _normalize(self, X):
        coefs = np.maximum(np.sqrt(np.square(X).sum(axis = 1)), self.K *
                np.ones(len(X)))
        return np.array(X) / np.array(coefs)[:,None]

    def _enforce_norm(self, X):
        """Ensure that X respects norm bounds

        Throw an error if any row of X has a norm bigger than K.

        NB: This method violates differential privacy, and should not be used
        indiscriminately with private data.

        """
        max_norm = np.sqrt(np.square(X).sum(axis = 1).max())
        if max_norm > self.K:
            raise ValueError("The l2 norm of the rows of X must be bounded by K = %f; "
                             "the maximum was %f" % (self.K, max_norm))

    def fit(self, X, y):
        """Fit model to features X and labels y.

        Normalizes the features in X to stay within the bound on the norm K: If
        the norm of a row is greater than K, normalize that row; otherwise,
        leave it intact.

        """
        if len(X) == 0:
            raise ValueError("Must provide non-empty X")

        X = self._normalize(X)

        self._enforce_norm(X) # This should never throw an error after the call to _normalize

        self.logit = self.logit.fit(X, y)

        # Split the privacy budget among each of the models fitted for each class

        e = self.epsilon / self.logit.coef_.shape[0]

        n = self.logit.coef_.shape[1]
        if self.logit.fit_intercept:
            n = n + 1

        for i in range(self.logit.coef_.shape[0]):
            noise = dp.laplacian_l2(e, n = n, \
                                    sensitivity = 2 * self.K_eff * self.logit.C / len(X))

            if self.logit.fit_intercept:
                self.logit.coef_[i] = self.logit.coef_[i] + noise[:-1]
                self.logit.intercept_[i] = self.logit.intercept_[i] + noise[-1]
            else:
                self.logit.coef_[i] = self.logit.coef_[i] + noise

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
        X = self._normalize(X)
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
        X = self._normalize(X)
        self._enforce_norm(X)
        return self.logit.score(X, y)

def test(epsilon, C, fit_intercept):
    X = pd.get_dummies(ttt.features)
    y = ttt.label
    K = np.sqrt(np.square(X).sum(axis = 1)).max()
    plogit = DPLogisticRegression(epsilon = epsilon, K = K, C = C, fit_intercept = fit_intercept)
    plogit = plogit.fit(X, y)
    print(plogit.score(X, y))
    return plogit

def test_multiclass(epsilon, C, fit_intercept):
    X = pd.get_dummies(nurs.features)
    y = nurs.label
    K = np.sqrt(np.square(X).sum(axis = 1)).max()
    plogit = DPLogisticRegression(epsilon = epsilon, K = K, C = C, fit_intercept = fit_intercept)
    plogit = plogit.fit(X, y)
    print(plogit.score(X, y))
    return plogit

class LogRegressionChooser:
    def __init__(hyperparams, mfeatures, train_dbs, scorefunc):
        pass
    def choose(DB):
        pass

    def fitPrivateModel(DB, epsilon):
        #Let's make this first
        pass
