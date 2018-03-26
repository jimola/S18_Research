import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#returns laplacian noise that should be added given sensitivity and epsilon
def laplacian(epsilon, n=1, sensitivity=1):
    lam = epsilon/sensitivity
    sign = 1-2*np.random.randint(0, 2, n)
    return np.random.exponential(1/lam, n) * sign

def hist_noiser(vals, epsilon=0):
    vals = np.array(vals, ndmin=1)
    if(epsilon == 0):
        return vals
    n = len(vals)
    fuzz = vals + laplacian(epsilon, n)
    count = (fuzz < 0).sum()
    while(count > 0):
        fuzz[fuzz < 0] = laplacian(epsilon, count)
        count = (fuzz < 0).sum()
    return fuzz

def exp_mech(utils, eps, sens):
    utils = np.array(utils, ndmin=1)
    if(eps == 0):
        return utils.argmax()
    utils= utils-max(utils)
    n = len(utils)
    weights = np.exp(eps*utils / (2*sens))
    prob = weights / sum(weights)
    u = np.random.rand()
    return prob.cumsum().searchsorted(u)

class ConditionalEntropy:
    def __init__(self, nrow):
        self.sens = (np.log(nrow)+1)/np.log(2)
    def get_ent(self, cnts):
        p = cnts / sum(cnts)
        ent = p*np.log(p) / np.log(2)
        return(sum(ent))
    def eval(self, r1, r2):
        parts = map(lambda x: len(x) * self.get_ent(pd.value_counts(r2[x])),
                r2.groupby(r1).groups.values())
        return sum(parts)

