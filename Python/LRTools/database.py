import collections
import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
class DB:
    """ Database class. Eases keeping track of epsilon, X, and y.
        params:
        X - database independent attributes
        y - dependent variable
        epsilon - value of epsilon for privacy
    """
    def __init__(self, X, y, epsilon=1, add_const=True):
        self.epsilon = epsilon
        self.ncol = X.shape[1]
        self.X = pd.get_dummies(X)
        if add_const:
            self.X['const'] = np.ones(self.X.shape[0])
        self.y = y

    @classmethod
    def dummy_vals(cls, nrow, ncol, nclasses, epsilon=1):
        X_dummy = np.random.normal(size=(nrow, ncol))
        y_dummy = pd.Series( X_dummy[:, 1] )
        y_dummy = (y_dummy < 0).astype(int)
        X_dummy = KBinsDiscretizer(nclasses,
                encode='ordinal').fit_transform(X_dummy)
        X_dummy = pd.DataFrame(X_dummy)
        return cls(X_dummy, y_dummy, epsilon)
    def get_norm(self):
        #X = pd.get_dummies(X)
        return np.linalg.norm(self.X, axis=1).max()

    def drop(self, thresh):
        self.X = self.X.loc[:, (self.X == 0).sum() / self.X.shape[0] < (1-thresh)]
        return self

    def normalize(self):
        X = self.X
        for c in X.columns:
            if X[c].min() >= 0:
                X[c] = X[c] / X[c].max()
            elif X[c].max() <= 0:
                X[c] = -X[c] / X[c].min()
            else:
                rnge = X[c].max() - X[c].min()
                t = (X[c] - X[c].min()) / rnge
                X[c] = 2*t-1
        return self

"""Metafeatures used.  Note that the size of the
dataset is considered public; the adjacency relation
allows changing the value of a record, but not deleting it."""
class DBMetafeatures:
    def __init__(self):
        self.sensitivities = collections.OrderedDict((
                ('nrow', 0), ('ncol', 0), ('eps', 0), ('numy', 1)
                ))
    
    def __call__(self, dataset):
        return collections.OrderedDict((
                ('nrow', dataset.X.shape[0]),
                ('ncol', dataset.ncol),
                ('eps', dataset.epsilon), 
                ('numy', dataset.y.sum()) ))

class DBSlicer:
    @staticmethod
    def reshape_dset(db, ncol, nrow, y_ratio, seed=12345, prng=None):
        """Rescales an input database to the desired parameters.
        Parameters
        ----------
        db: Input array. We assume last col is the output col.
            The output col must be binary.
            
        ncol: Desired number of columns in output.
        
        nrow: Desired number of rows.
        
        y_ratio: Desired percentage of class 2 in the output
        
        seed: seed value to use. Default: 12345
        
        prng: random number generator. One of seed or prng must not be None. 
        """
        if(prng == None):
            prng = np.random.RandomState(seed)
        ys = db[db.columns[-1]]
        v1, v2 = ys.unique()[:2]
        Z1 = db[ys == v1]
        s1 = int(y_ratio*nrow)
        Z2 = db[ys == v2]
        s2 = nrow - s1
        def reshape(Z, nrow):
            db = pd.DataFrame()
            while db.shape[0] + Z.shape[0] < nrow:
                db = pd.concat((db, Z), ignore_index=True)
            return pd.concat((db, Z.sample(nrow - db.shape[0])), ignore_index=True)
        db = pd.concat((reshape(Z1, s1), reshape(Z2, s2)), ignore_index=True)
        db_x = db[db.columns[:-1]]
        ys = db[db.columns[-1]]
        rand_mat = prng.normal(0, 1, (db_x.shape[1], ncol))
        db_x = db_x.dot(rand_mat)
        return pd.concat((db_x, ys), axis=1, ignore_index=True).sample(frac=1)
    
