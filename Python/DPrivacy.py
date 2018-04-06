import numpy as np
import pandas as pd

#Class representing a Database
class Database:
    def __init__(self, train, test, x_names, y_name):
        self.train=train
        self.test=test
        self.y_name=y_name
        self.x_names=x_names
    @classmethod
    def from_dataframe(cls, d, y_idx=-1, cutoff=0.7):
        for x in d.columns:
            d[x] = np.unique(d[x], return_inverse=True)[1]
            d[x] = d[x].astype('category')
        d = d.reindex(np.random.permutation(d.index))
        cutoff = int(cutoff*len(d))
        y_name = d.columns[y_idx]
        x_names = d.columns[d.columns != y_name]
        return cls(d[:cutoff], d[cutoff:], x_names, y_name)

#returns laplacian noise that should be added given sensitivity and epsilon
def laplacian(epsilon, n=1, sensitivity=1):
    lam = epsilon/sensitivity
    sign = 1-2*np.random.randint(0, 2, n)
    return np.random.exponential(1/lam, n) * sign

def hist_noiser(vals, epsilon=0):
    if(isinstance(vals, int)):
        vals = np.array(vals, ndmin=1)
    if(epsilon == 0):
        return vals
    n = len(vals)
    fuzz = vals + laplacian(epsilon, n)
    """
    count = (fuzz < 0).sum()
    while(count > 0):
        fuzz[fuzz < 0] = laplacian(epsilon, count)
        count = (fuzz < 0).sum()
    """
    return fuzz

def exp_mech(utils, eps, sens):
    utils = np.array(utils, ndmin=1)
    if(eps == 0):
        return utils.argmax()
    utils = utils-max(utils)
    n = len(utils)
    weights = np.exp(eps*utils / (2*sens))
    prob = weights / sum(weights)
    u = np.random.rand()
    return prob.cumsum().searchsorted(u)

class ConditionalEntropy:
    def __init__(self, nrow):
        self.sens = (np.log(nrow)+1)/np.log(2)
    def get_ent(self, cnts):
        cnts = cnts[cnts > 0]
        p = cnts / sum(cnts)
        ent = p*np.log(p) / np.log(2)
        return(sum(ent))
    def eval(self, r1, r2):
        parts = map(lambda x: len(x) * self.get_ent(pd.value_counts(r2[x])),
                r2.groupby(r1).groups.values())
        return sum(parts)

