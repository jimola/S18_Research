
from sklearn import model_selection, feature_selection
import collections
import numpy as np
import pandas as pd
import DPrivacy as dp
class DB:
    def __init__(self, X, y, X_test, y_test, epsilon=1, depth=0, max_depth=0):
        self.epsilon = epsilon
        self.ncol = X.shape[1]
        self.X = X
        self.y = y
        self.X_test = X_test
        self.y_test = y_test
        self.depth = depth
        self.max_depth = max_depth
        #branching_factor?
        
class DBMetas:
    def __init__(self):
        self.sensitivities = collections.OrderedDict((
        ('nrow', 1), ('ncol', 0), ('eps', 0), ('depth', 0), ('C', 0), ('bf', 0), ('t', 0)
                                                    ))
    
    def __call__(self, dataset):
        att_sizes = np.array([len(dataset.X[c].cat.categories) for c in dataset.X])
        return collections.OrderedDict((
                ('nrow', dataset.X.shape[0]), 
                ('ncol', dataset.X.shape[1]),
                ('eps', dataset.epsilon),
                ('depth', dataset.depth),
                ('C', len(dataset.y.cat.categories)),
                ('bf', att_sizes.mean()),
                ('t', att_sizes.max()) ))

def entropy(y):
    arr = pd.value_counts(y)
    arr = arr[arr > 0]
    arr = arr/arr.sum()
    return (-np.log2(arr) * arr).sum()

class TreePart:
    def __init__(self, splits=5):
        self.splits=splits
        self.kf = model_selection.KFold(splits)
        self.lo = model_selection.LeaveOneOut()
        self.numruns = 0
    def get_expected_correct(self, y, epsilon):
        if len(y) == 0:
            return 0
        tot_correct = 0
        if len(y) < self.splits:
            gen = self.lo.split(y)
        else:
            gen = self.kf.split(y)
        for train_idx, test_idx in gen:
            hist = pd.value_counts(y.iloc[train_idx])
            noisy_hist = dp.hist_noiser(hist, epsilon)
            pred = noisy_hist.idxmax()
            tot_correct += (y.iloc[test_idx] == pred).sum()
        return tot_correct / len(y)
    
"""The leaf algorithm"""
class Leaf(TreePart):
    def error(self, db):
        if len(db.y) == 0:
            return 0
        return 1.0-self.get_expected_correct(db.y, db.epsilon)
    def run(self, db):
        self.numruns += 1
        frequencies = pd.value_counts(db.y)
        noisy_freqs = dp.hist_noiser(frequencies, db.epsilon)
        return np.repeat(noisy_freqs.idxmax(), db.y_test.size)
    
"""The branch algorithm (splits the data)"""
class Split(TreePart):
    def error(self, db):
        if len(db.y) == 0:
            return 0
        corrects = []
        probs = []
        for col in db.X.columns:
            x = db.X[col]
            cats = x.cat.categories
            correct = 0
            tot_ent = 0
            for cat in cats:
                leaf_correct = self.get_expected_correct(db.y[x == cat], db.epsilon)
                correct += leaf_correct * (x == cat).sum()
                ent = entropy(db.y[x == cat])
                tot_ent += ent*(x==cat).sum()
            correct /= len(db.y)
            corrects.append(correct)
            probs.append(tot_ent)
        C = np.array(corrects)
        probs = np.array(probs)
        probs -= probs.min()
        D = np.exp(-probs*db.epsilon/13) #13 is an upper bound on the 
                                         #sensitivity of entropy, log_2 |X|
        probs = D/D.sum()
        return 1-probs.dot(C)
    def run(self, db):
        self.numruns += 1
        return None

"""Private Decision Tree algorithm"""
class PDTree:
    def __init__(self):
        self.leaf = Leaf()
        pass
    
    def entropy(self, y):
        arr = pd.value_counts(y)
        arr = arr[arr > 0]
        arr = arr/arr.sum()
        return (-np.log2(arr) * arr).sum()
    
    def decision_helper(self, db, cm):
        if db.depth == db.max_depth:
            return self.leaf.run(db)
        action = cm.choose(db)
        if action is not None:
            return action
        utils = []
        for col in db.X.columns:
            x = db.X[col]
            cats = x.cat.categories
            cur_ent = 0
            for cat in cats:
                ent = self.entropy(db.y[x == cat])
                cur_ent += ent * (x == cat).sum()
            #cur_ent /= len(db.y)
            utils.append(-cur_ent)
        best_idx = dp.exp_mech(utils, db.epsilon, 13) #Change sensitivity
        col_name = db.X.columns[best_idx]
        new_cols = db.X.columns[db.X.columns != col_name]
        splitX = db.X[col_name]
        splitX_test = db.X_test[col_name]
        preds = np.repeat(db.y.cat.categories[0], len(db.y_test))
        for att in splitX_test.unique():
            train_split = db.X.loc[splitX == att, new_cols]
            y_split = db.y[splitX == att]
            test_split_loc = db.X_test[col_name] == att 
            test_split = db.X_test.loc[test_split_loc, new_cols]
            test_split_y = db.y_test.loc[test_split_loc]
            if(test_split_y.size > 0):
                db_new = DB(train_split, y_split, test_split, 
                           test_split_y, db.epsilon, db.depth+1, db.max_depth)
                preds[test_split_loc] = self.decision_helper(db_new, cm)
        return preds
    
    def fit_and_predict(self, data, cm):
        budget = data.epsilon / data.X.shape[1]
        data.epsilon = budget
        return self.decision_helper(data, cm)

class CoefCM:
    def __init__(self, coefs, const):
        self.m = DBMetas()
        self.coefs = np.array(coefs)
        self.const = const
        self.leaf = Leaf()
        self.split = Split()
    def choose(self, data, ratio=0.3):
        budget = data.epsilon*ratio
        metas = self.m(data)
        metas['nrow'] += dp.laplacian(budget, sensitivity=1)[0]
        data.epsilon -= budget
        metas = np.array(list(metas.values()))
        metas = np.log(np.maximum(metas, 1))
        if metas.dot(self.coefs) <= self.const:
            return self.leaf.run(data)
        else:
            return self.split.run(data)
