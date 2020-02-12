import pandas as pd
import numpy as np
import DPrivacy as dp
import graphviz
import copy
import sys
sys.path = ['./scikit-learn/build/lib.linux-x86_64-3.6/'] + sys.path
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split

# TODO
# - Replace the metafeature class with a callable object.
# - Perhaps we should rename error to score, which is more general.

class MFTransformation:
    """ Represents a transformation on metafeatures"""
    def __call__(self, BaseX):
        pass

class ArithMFTransformation(MFTransformation):
    def __init__(self, coefs):
        self.coefs = copy.copy(coefs)
    def __call__(self, BaseX):
        ans = np.zeros(BaseX.shape[0])
        for i in range(BaseX.shape[1]):
            c = BaseX.columns[i]
            ans += BaseX[c] * self.coefs[i]
        return ans
class MetaFeatureHelper:
    @staticmethod
    def get_all_trans(size):
        trans = []
        coefs = [-1 for x in range(size)]
        while True:
            idx = 0
            while idx < size and coefs[idx] == 0:
                idx += 1
            if idx < len(coefs) and coefs[idx] == -1:
                trans.append(ArithMFTransformation(coefs))
            idx = size-1
            while idx > -1:
                if coefs[idx] == 1:
                    coefs[idx] = -1
                else:
                    coefs[idx] += 1
                    break
                idx -= 1
            if idx == -1:
                break
        return trans
class DTChoice:
    """Choice maker based on sklearn decision trees

    Parameters
    ----------

    train_set: A list of public databases

    mfs: A callable object for computing metafeatures on databases.  The
    returned metafeatures must be a dictionary object mapping metafeature names
    to their values. The mfs object must have a sensitivity attribute, with is a
    dictionary mapping metafeature names to their sensitivities.

    algs: A dictionary mapping names to algorithms. Each algorithm must implement
    a run method, which executes the algorithm on a database, and an error
    method, which computes the algorithm's error on a database.
    
    regrets: If the regrets are already known, supply them here. Will avoid the
    training phase of the CM.

    C: Value of C used to train the special regret-based decision tree

    trans: feature transformations to use. Best to be kept as 'default'
    """

    def __init__(self, train_set, mfs, algs, regrets=None, C=0,
            trans='default', verbose=0):
        self.metafeatures = mfs
        self.algs = algs
        if isinstance(train_set, pd.DataFrame):
            self.X = train_set
        else:
            self.X = pd.DataFrame([mfs(t) for t in train_set])
        self.C = C
        usage = np.array(list(mfs.sensitivities.values()))
        usage[usage > 0] = 1
        self.is_used = usage
        if regrets is None:
            if verbose > 0:
                checkpoint = int( len(train_set) / verbose)
                checkpoint = max(1, checkpoint)
            else:
                checkpoint = len(train_set) + 1
            regrets = []
            for i, t in enumerate(train_set):
                regrets.append({name: alg.error(t) for name, alg in
                    algs.items()})
                if i % checkpoint == 0:
                    print("%0.1f%%" % (100*i / len(train_set)))
            self.regrets = pd.DataFrame(regrets)
        else:
            self.regrets = regrets
        if trans == 'default':
            self.trans = MetaFeatureHelper.get_all_trans(self.X.shape[1])
        else:
            self.trans = []
        log_X = np.log(np.maximum(1e-8, self.X))
        self.T = pd.DataFrame([t(log_X) for t in
            self.trans]).reset_index(drop=True).T.reset_index(drop=True)

        self.y = self.regrets.idxmin(axis=1)
        self.model = DecisionTreeClassifier()
        self.retrain_model()
    
    #Change metafeatures
    def update_metas(self, train_set, mfs):
        self.X = pd.DataFrame([mfs(t) for t in train_set])
        self.T = pd.DataFrame([t(self.X) for t in
            self.trans]).reset_index(drop=True).T.reset_index(drop=True)
        usage = np.array(list(mfs.sensitivities.values()))
        usage[usage > 0] = 1
        self.is_used = usage
        self.retrain_model()

    #Helper method
    def retrain_model(self):
        X = pd.concat((self.X, self.T), axis=1)
        self.model.fit(X, self.y, self.regrets, self.C)

    #Return the label of the best algorithm.
    def get_best_alg(self, data, budget):
        sens = self.metafeatures.sensitivities
        nnz = np.count_nonzero(self.is_used)
        feature_budget = budget / nnz
        X = self.metafeatures(data)
        #noisy_X = pd.DataFrame([{name: value + np.random.laplace(0, sens[name]/
        #                             feature_budget)
        #                         for name, value in self.metafeatures(data).items()}])
        noisy_X = pd.DataFrame(self.metafeatures(data), index=[0]) \
                + pd.DataFrame(self.metafeatures.sensitivities, index=[0]) \
                              .apply(lambda x: np.random.laplace(0,
                                  x/feature_budget))
        log_noisy_X = np.log(np.maximum(1e-8, noisy_X))
        noisy_T = pd.DataFrame([t(log_noisy_X) for t in
            self.trans]).reset_index(drop=True).T

        X = pd.concat((noisy_X, noisy_T), axis=1)
        S = self.X.shape[1]
        used = np.zeros(S)
        node_counts = self.model.decision_path(X).data-1 
        U = np.unique(node_counts[:-1])
        used[U[U < S]] = 1
        U = U[U >= S] - S
        used += np.any([self.trans[i].coefs for i in U], axis=0)
        used = used > 0
        nfeature_used = self.is_used.dot(used)
        alg = self.model.predict(X)[0]
        return alg, nfeature_used * feature_budget

    #Choose and run the best algorithm in a DP way
    def choose(self, data, ratio=0.3):
        budget = data.epsilon*ratio
        tot_eps = data.epsilon
        data.epsilon -= budget
        (best, used) = self.get_best_alg(data, budget)
        data.epsilon = tot_eps - used
        return self.algs[best].run(data)

    def get_errors(self, data, ratio=0.3):
        #data = copy.copy(data)
        budget = data.epsilon*ratio
        errors = pd.DataFrame([{name: alg.error(data)
                                for name, alg in self.algs.items()}])

        (best, used) = self.get_best_alg(data, budget)
        best_alg = self.algs[best]
        data.epsilon = data.epsilon - used
        R = best_alg.error(data)
        errors['cm'] = R
        return errors 

    def get_approximate_regret(self, return_std=False, test_ratio=0.3):
        """
        Splits data into training and test and returns average regrets on the
        test split for each algorithm and for this DTChoice object.

        The DTChoice regret is approximate (and an underestimate) for two 
        reasons. Let A = ratio*epsilon and B = (1-ratio)*epsilon.

        First, we don't add Laplace(A) noise to the metafeatures when we 
        predict on them.

        Second, the algorithm we choose isn't run with B budget---it's run with
        epsilon budget instead.

        """
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                self.y, test_size=test_ratio)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train.idxmin(axis=1))
        algs = model.predict(X_test)
        perfs = y_test.lookup(y_test.index, algs)
        R = np.concatenate((np.array(y_test), perfs[:, None]), axis=1)
        R = R - np.min(R, axis=1)[:, None]
        if(return_std):
            return (R.mean(axis=0), R.std(axis=0))
        else:
            return R.mean(axis=0)

    def print_tree(self, of=None):
        dot_data = export_graphviz(self.model, out_file=of, filled=True,
                rounded=True)
        graph = graphviz.Source(dot_data)
        return graph

    def print_arith_coef(self, idx):
        coefs = self.trans[idx].coefs
        L = list(self.metafeatures.sensitivities.keys())
        top = []
        bot = []
        for i in range(len(L)):
            if coefs[i] == 1:
                top.append(L[i])
            elif coefs[i] == -1:
                bot.append(L[i])
        return '*'.join(top) + ' / ' + '*'.join(bot)
