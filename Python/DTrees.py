import DPrivacy
import numpy as np
import pandas as pd
class Controller:
    def __init__(self, db, max_depth, budget, nt=1):
        self.max_depth = max_depth
        self.db = db
        self.budget = budget/nt
        self.init_numcols = len(db.x_names)
        self.nt = nt
    def leaf(self, db, eps):
        freq_table = pd.value_counts(db.train[db.y_name])
        noisy_counts = DPrivacy.hist_noiser(freq_table, eps)
        return freq_table.keys()[noisy_counts.argmax()]

    def decision_helper(self, db, eps):
        if(len(db.test) == 0):
            return np.array([])
        (used, attr) = self.get_action(db)
        eps -= used
        if(attr == None):
            pred = self.leaf(db, eps)
            return np.repeat(pred, len(db.test))
        else:
            new_x = db.x_names[db.x_names != attr]
            preds = np.repeat(db.train[db.y_name].iloc[0], len(db.test))
            for cat in iter(np.unique(db.train[attr])):
                train_split = db.train[db.train[attr] == cat]
                test_split_loc = db.test[attr] == cat
                db_new = DPrivacy.Database(
                        train_split, db.test[test_split_loc], new_x, db.y_name)
                preds[test_split_loc] = self.decision_helper(db_new, eps)
            return preds
    
    #Used for collecting runtime information
    def collect_info_helper(self, db, eps):
        if(len(db.test) == 0):
            return np.array([0, 0, 0])
        (used, attr) = self.get_action(db)
        eps -= used
        if(attr == None):
            pred = self.leaf(db, eps)
            freqs = pd.value_counts(db.test[db.y_name])
            if(pred in freqs):
                return np.array([freqs[pred], 0, 1])
            else:
                return np.array([0, 0, 1])
        else:
            new_x = db.x_names[db.x_names != attr]
            ret = np.array([0, 0, 1])
            for cat in iter(np.unique(db.train[attr])):
                train_split = db.train[db.train[attr] == cat]
                test_split = db.test[db.test[attr] == cat]
                db_new = DPrivacy.Database(
                        train_split, test_split, new_x, db.y_name)
                ret += self.collect_info(db_new, eps)
            self_leaf_cnt = pd.value_counts(db.test[db.y_name]).max()
            ret[1] += int(self_leaf_cnt > ret[0])
            return ret
    def collect_info(self):
        return collect_info_helper(self, self.db, self.budget)
    def get_preds(self):
        return get_decision_helper(self, self.db, self.budget)
    def get_accuracy(self):
        preds = self.get_preds()
        c = sum(preds == self.db.test[self.db.y_name])
        return c / len(self.db.test)

class NonPrivate(Controller):
    def __init__(self, db, max_depth):
        Controller.__init__(self, db, max_depth, 0)
        self.util_func = DPrivacy.ConditionalEntropy(len(db.train))
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        y = db.train[db.y_name]
        if(depth >= self.max_depth or len(y.unique()) <= 1):
            return(0, None)
        utils = list(map(lambda x: self.util_func.eval(db.train[x], y), db.x_names))
        idx = DPrivacy.exp_mech(utils, 0, None)
        return (0, db.x_names[idx])

def get_gini(sample, num_zeros=0):
    tot_size = num_zeros + len(sample)
    B = np.sort(sample).cumsum().sum() / sample.sum()
    A = (tot_size+1)/2
    return (A-B) / A

#How important is it to measure the size of the dataset?
#We spend epsilon budget to get info on slice_size budget
def get_size_gini(db, eps, slice_size=4):
    cols = list(np.random.choice(db.x_names, slice_size, False))
    attr_sizes = list(map(lambda x: len(np.unique(db.train[x])), cols))
    prods = np.array(attr_sizes).prod()
    sizes = np.array(db.train.groupby(cols)[db.y_name].count())
    sizes = DPrivacy.hist_noiser(sizes, 0.75*eps)
    num_zeros = int(DPrivacy.hist_noiser(prods - len(sizes), 0.25*eps))
    skew = sizes / len(db.train)
    return get_gini(skew, num_zeros)

def get_density(db):
    b = list(map(lambda x: len(db.train[x].unique()), db.x_names))
    r = len(db.train) / np.array(b).prod()
    return np.tanh(r/2)

def get_features(db, eps):
    B = min(1, 0.1*eps)
    feats = {'density': [get_density(db)], 'disuniformity': [get_size_gini(db, B)]}
    return (pd.DataFrame(feats), B)

#Friedman and Schuster
class FS(Controller):
    def __init__(self, db, budget, max_depth):
        Controller.__init__(self, db, max_depth, budget)
        self.calc_budget = budget / (2*(max_depth+1))
        col_sizes = db.train.apply(lambda x: len(x.unique()))
        t = max(col_sizes[db.x_names])
        self.stop_const = np.sqrt(2)*t*col_sizes[db.y_name] / self.calc_budget
        self.util_func = DPrivacy.ConditionalEntropy(len(db.train))
        self.name = 'Friedman and Schuster'
    #return a number from 0 to 1
    def eval_annotation(self, feat):
        return float(feat.disuniformity*feat.density)
    def get_action(self, db):
        nrow = DPrivacy.hist_noiser(len(db.train), self.calc_budget)
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth or nrow < self.stop_const):
            return (self.calc_budget, None)
        y = db.train[db.y_name]
        utils = list(map(lambda x: self.util_func.eval(db.train[x], y), db.x_names))
        idx = DPrivacy.exp_mech(utils, self.calc_budget, self.util_func.sens)
        return (2*self.calc_budget, db.x_names[idx])

#Mohammed et al.
class MA(Controller):
    def __init__(self, db, budget, max_depth):
        Controller.__init__(self, db, max_depth, budget)
        self.calc_budget = budget/(max_depth+1)
        self.util_func = DPrivacy.ConditionalEntropy(len(db.train))
        self.name = 'Mohammed et al.'
    def eval_annotation(self, feat):
        return float((1-feat.disuniformity)*feat.density)
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth):
            return (0, None)
        y = db.train[db.y_name]
        utils = list(map(lambda x: self.util_func.eval(db.train[x], y), db.x_names))
        idx = DPrivacy.exp_mech(utils, self.calc_budget, self.util_func.sens)
        return (self.calc_budget, db.x_names[idx])

#Jagannathan et al.
#We force different attributes at beginning
class Jag(Controller):
    def __init__(self, db, budget, nt):
        k = len(db.x_names)
        b = sum(map(lambda x: len(np.unique(db.train[x])), db.x_names))
        b = b / k
        max_dep = min([np.log(len(db.train))/np.log(b)-1, k/2])
        Controller.__init__(self, db, max_dep, budget, nt)
        self.name = 'Jagannathan et al.'
        self.starters = np.random.choice(db.x_names, nt, False)
        self.treecnt = 0
    def eval_annotation(self, feat):
        dens = float(feat.density)
        return max(1-3*dens, 0.33*(1-dens))
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth):
            return(0, None)
        if(depth == 0):
            return (0, self.starters[self.treecnt])
        idx = int(np.random.uniform() * len(db.x_names))
        return (0, db.x_names[idx])
    def get_preds(self):
        self.treecnt = 0
        nrow = len(self.db.test)
        L = []
        for i in range(0, self.nt):
            L.append(self.decision_helper(self.db, self.budget))
            self.treecnt+=1
        L = np.array(L)
        U = np.unique(L)
        pred = np.repeat(U[0], nrow)
        freq = np.repeat(0, nrow)
        for clss in U:
            freq_clss = np.repeat(0, nrow)
            for row in L:
                freq_clss += row == clss
            mask = freq_clss > freq
            pred[mask] = clss
            freq[mask] = freq_clss[mask]
        return pred

#Fletcher and Islam
#Work In Progress
class FI(Controller):
    def __init__(self, db, budget, nt):
        Controller.__init__(self, db, 0, budget, nt)
        self.name = 'Fletcher and Islam'
    def decision_helper(self, db, eps):
        if(len(db.test) == 0):
            return np.array([])
        (used, attr) = self.get_action(db)
        eps -= used
        if(attr == None):
            pred = self.leaf(db, eps)
            return np.repeat(pred, len(db.test))
        else:
            new_x = db.x_names[db.x_names != attr]
            preds = np.repeat(db.train[db.y_name].iloc[0], len(db.test))
            for cat in iter(np.unique(db.train[attr])):
                train_split = db.train[db.train[attr] == cat]
                test_split_loc = db.test[attr] == cat
                db_new = DPrivacy.Database(
                        train_split, db.test[test_split_loc], new_x, db.y_name)
                preds[test_split_loc] = self.decision_helper(db_new, eps)
            return preds
    def eval_annotation(self, feat):
        pass
    def get_action(self, db):
        pass
#TODO: Implement other algos

#TODO: Features should be used to help set algorithmic parameters
class ChoiceMaker:
    def __init__(self, db, budget, nt, depth):
        (feats, used) = get_features(db, budget)
        budget -= used
        alglist = [FS(db, budget, 5), MA(db, budget, 5), Jag(db, budget, nt)]
        perfs = np.array(list(map(lambda x: x.eval_annotation(feats), alglist) ))
        self.alg = alglist[perfs.argmax()]
    def get_accuracy(self):
        return self.alg.get_accuracy()

#import cProfile
#ma = MA(dblist['bind'], 5, 5,)
#cProfile.run('ma.get_accuracy()')
#Email Justin for a Skype
