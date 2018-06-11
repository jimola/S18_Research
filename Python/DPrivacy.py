import numpy as np
import pandas as pd

class Database:
    """Class of databases split into training and test points

    Parameters
    ----------

    train : The training set

    test : The test set

    x_names : Columns that contain the features

    y_name : Column that contains the target variable
    """

    def __init__(self, train, test, x_names, y_name):
        self.train=train
        self.test=test
        self.y_name=y_name
        self.x_names=x_names

    @classmethod
    def from_dataframe(cls, d, y_idx=-1, cutoff=0.7):
        """Create a database from a Pandas dataframe.

        Parameters
        ----------

        d : The dataframe

        y_idx : The name of the column corresponding to the target variable
        (default -1)

        cutoff : The proportion of rows that go into the train set (default 0.7)

        """
        # We could maybe use sklearn.model_selection.train_test_split for this.
        for x in d.columns:
            if(d[x].dtype == 'O'):
                d[x] = np.unique(d[x], return_inverse=True)[1]
                d[x] = d[x].astype('category')
        d = d.reindex(np.random.permutation(d.index))
        cutoff = int(cutoff*len(d))
        y_name = d.columns[y_idx]
        x_names = d.columns[d.columns != y_name]
        return cls(d[:cutoff], d[cutoff:], x_names, y_name)

def laplacian(epsilon, n=1, sensitivity=1):
    """Laplace mechanism

    Parameters
    ----------

    epsilon : The sought privacy level

    n : The number of dimensions

    sensitivity : The sensitivity of the query that the noise will be applied
        to, measured with respect to the l1 norm

    Returns
    -------

    out : A vector drawn from the Laplace distribution
    """
    if sensitivity == 0:
        return 0.0
    else:
        lam = epsilon/sensitivity
        sign = 1-2*np.random.randint(0, 2, n)
        return np.random.exponential(1/lam, n) * sign

def sampleR(d, beta):
    """Sample r in R^d proportionally to exp(|r|_2 / beta)"""
    erlang = sum(np.random.exponential(scale = beta, size = d))
    direction = np.random.normal(size = d)
    direction = direction / np.linalg.norm(direction)
    return erlang * direction

def laplacian_l2(epsilon, n = 1, sensitivity = 1):
    """Return noise to achieve differential privacy with l2 sensitivity

    Parameters
    ----------

    epsilon : float
        The privacy budget. Must be greater than 0.

    n : int
        The dimension of the noise vector to be generated (default 1)

    sensitivity : float
        The sensitivity of the result that the noise will be applied to,
        measured with respect to the l2 norm (default 1)
    """
    if sensitivity == 0:
        return 0
    else:
        return sampleR(d = n, beta = sensitivity / epsilon)

def hist_noiser(vals, epsilon=0):
    """Apply Laplace noise to a histogram

    Parameters
    ----------

    vals : The histogram; a vector of counts for each bucket

    epsilon : The privacy parameter

    Returns
    -------

    out : The noised histogram
    """
    if(np.isscalar(vals)):
        vals = np.array(vals, ndmin=1)
    if(epsilon == 0):
        return vals
    n = len(vals)
    fuzz = vals + laplacian(epsilon, n)
    # count = (fuzz < 0).sum()
    # while(count > 0):
    #     fuzz[fuzz < 0] = laplacian(epsilon, count)
    #     count = (fuzz < 0).sum()
    return fuzz

def exp_mech(utils, eps, sens):
    """Exponential mechanism

    Parameters
    ----------

    utils : The vector of utilities for each element

    eps : The privacy parameter

    sens : The sensitivity of the utility function


    Returns
    -------

    out : The chosen element, represented as an integer in range(len(utils))
    """
    utils = np.array(utils, ndmin=1)
    if(eps == 0):
        return utils.argmax()
    utils = utils-max(utils)
    weights = np.exp(eps*utils / (2*sens))
    prob = weights / sum(weights)
    u = np.random.rand()
    return prob.cumsum().searchsorted(u)


class ConditionalEntropy:
    """Conditional entropy of a dataset

    Parameters
    ----------

    nrow : Integer
        The size of the data set on which to perform this computation.  Since
        this size determines the sensitivity of the computation, you must ensure
        that each instance is only applied to data sets of the corresponding size.

    """

    def __init__(self, nrow):
        self.sens = (np.log(nrow)+1)/np.log(2)

    def get_ent(self, cnts):
        """Entropy of a histogram

        Parameters
        ----------

        cnts : Pandas Series object
            Histogram of values


        Returns
        -------

        out : Float
            The entropy of the corresponding probability distribution
        """
        cnts = cnts[cnts > 0]
        p = cnts / sum(cnts)
        ent = p*np.log(p) / np.log(2)
        return(sum(ent))

    def eval(self, r1, r2):
        """Compute the conditional entropy.


        Parameters
        ----------

        r1 : Pandas Series object
            The conditioning attribute

        r2 : Pandas Series object
            The attribute of which to compute the entropy conditioned on r1.  Must
            have the same size as r1.

        """
        parts = [len(x) * self.get_ent(pd.value_counts(r2[x])) for x in
                r2.groupby(r1).groups.values()]
        return sum(parts)
