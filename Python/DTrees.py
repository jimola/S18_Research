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
        return noisy_counts.argmax()

    def decision_helper(self, db, eps):
        if(len(db.test) == 0):
            return np.array([])
        if(len(db.train) == 0):
            pred = np.random.choice(db.train[db.y_name].cat.categories)
            return np.repeat(pred, len(db.test))
        (used, col_name) = self.get_action(db)
        eps -= used
        if(col_name == None):
            pred = self.leaf(db, eps)
            return np.repeat(pred, len(db.test))
        else:
            new_x = db.x_names[db.x_names != col_name]
            preds = np.repeat(db.train[db.y_name].iloc[0], len(db.test))
            if(len(db.train) < 100):
                L = db.train[col_name].unique()
            else:
                L = db.train[col_name].cat.categories
            for att in L:
            #for att in db.train[col_name].unique():
                train_split = db.train[db.train[col_name] == att]
                test_split_loc = db.test[col_name] == att
                db_new = DPrivacy.Database(
                        train_split, db.test[test_split_loc], new_x, db.y_name)
                preds[test_split_loc] = self.decision_helper(db_new, eps)
            return preds
    
    def get_preds(self):
        return self.decision_helper(self.db, self.budget)
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
        utils = [self.util_func.eval(db.train[x], y) for x in db.x_names]
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
    attr_sizes = [len(db.train[x].unique()) for x in db.x_names]
    prods = np.array(attr_sizes).prod()
    sizes = np.array(db.train.groupby(cols)[db.y_name].count())
    sizes = DPrivacy.hist_noiser(sizes, 0.75*eps)
    num_zeros = int(DPrivacy.hist_noiser(prods - len(sizes), 0.25*eps))
    skew = sizes / len(db.train)
    return get_gini(skew, num_zeros)

def get_density(db):
    b = [len(db.train[x].unique()) for x in db.x_names]
    r = len(db.train) / np.array(b).prod()
    return np.tanh(r/2)

def get_features(db, eps):
    B = min(0.5, 0.1*eps)
    feats = {'density': [get_density(db)], 'disuniformity': [get_size_gini(db, B)]}
    return (pd.DataFrame(feats), B)

#Friedman and Schuster
class FS(Controller):
    def __init__(self, db, budget, max_depth):
        Controller.__init__(self, db, max_depth, budget)
        self.calc_budget = budget / (2*(max_depth+1))
        col_sizes = db.train.apply(lambda x: len(x.cat.categories))
        t = max(col_sizes[db.x_names])
        self.stop_const = np.sqrt(2)*t*col_sizes[db.y_name] / self.calc_budget
        self.util_func = DPrivacy.ConditionalEntropy(len(db.train))
        self.name = 'Friedman and Schuster'
    #Consistently affected in a small way by epsilon
    #Likes higher levels of disuniformity
    #Likes lower density
    def eval_annotation(self, feat):
        return float(feat.disuniformity*(1-feat.density))
    def get_action(self, db):
        nrow = DPrivacy.hist_noiser(len(db.train), self.calc_budget)
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth or nrow < self.stop_const):
            return (self.calc_budget, None)
        y = db.train[db.y_name]
        utils = [self.util_func.eval(db.train[x], y) for x in db.x_names]
        idx = DPrivacy.exp_mech(utils, self.calc_budget, self.util_func.sens)
        return (2*self.calc_budget, db.x_names[idx])

#Mohammed et al.
class MA(Controller):
    def __init__(self, db, budget, max_depth):
        k = len(db.x_names)
        b = sum([len(db.train[x].cat.categories) for x in db.x_names])
        b = b / k
        max_depth = min(max_depth, np.log(len(db.train))/np.log(b)-1, k/2)
        Controller.__init__(self, db, max_depth, budget)
        self.calc_budget = budget/(max_depth+1)
        self.util_func = DPrivacy.ConditionalEntropy(len(db.train))
        self.name = 'Mohammed et al.'
    #Sensitive to epsilon
    #Likes low levels of disuniformity. Very sensitive
    #Likes high density
    def eval_annotation(self, feat):
        return float(np.tanh(self.budget/2)*(1-feat.disuniformity)*(1+feat.density)/2)
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth):
            return (0, None)
        y = db.train[db.y_name]
        utils = [self.util_func.eval(db.train[x], y) for x in db.x_names]
        idx = DPrivacy.exp_mech(utils, self.calc_budget, self.util_func.sens)
        return (self.calc_budget, db.x_names[idx])

#Jagannathan et al.
#We force different attributes at beginning
class Jag(Controller):
    def __init__(self, db, budget, nt):
        k = len(db.x_names)
        b = sum([len(db.train[x].cat.categories) for x in db.x_names])
        b = b / k
        max_dep = min([np.log(len(db.train))/np.log(b)-1, k/2])
        Controller.__init__(self, db, max_dep, budget, nt)
        self.name = 'Jagannathan et al.'
        q = nt // len(db.x_names)
        r = nt % len(db.x_names)
        self.starters = list(db.x_names)*q+list(np.random.choice(db.x_names, r, False))
        self.treecnt = 0
    #Sensitive to very small values of epsilon when disuniformity is high
    #Likes low density
    #Likes epsilon to be reasonably high
    def eval_annotation(self, feat):
        dens = float(feat.density)
        return float(np.sqrt(1-dens)*feat.disuniformity*min(self.budget, 1.0))
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
        U = self.db.train[self.db.y_name].cat.categories
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
#This is a Work In Progress
#TODO: Fix this work in progress
class FI(Controller):
    def __init__(self, db, budget, nt):
        Controller.__init__(self, db, 0, budget, nt)
        self.C = len(db.train[db.y_name].cat.categories)
        self.noise_est = self.C*np.sqrt(2) / budget
        self.name = 'Fletcher and Islam'
    def leaf(self, db, eps):
        freq_table = pd.value_counts(db.train[db.y_name])
        noisy_counts = DPrivacy.hist_noiser(freq_table, eps)
        return (noisy_counts.argmax(), noisy_counts)
    #Return (preds, num_leaves, size_est)
    def decision_helper(self, db, eps, size_est):
        """
        if(len(db.test) == 0):
            return np.array([])
        """
        if(len(db.train) == 0):
            p = np.random.permutation(db.x_names)
            idx = 0
            num_leaves = 1
            while(idx < len(db.x_names) and size_est >= self.noise_est):
                A = len(db.train[p[idx]].cat.categories)
                size_est /= A
                idx += 1
                num_leaves *= A
            #Central Limit Theorem. Sum up num_leaves laplace vars with 2/eps^2 variance
            size = np.random.normal(0, np.sqrt(2*num_leaves)/eps, self.C)
            pred = np.random.choice(db.train[db.y_name].cat.categories)
            return (np.repeat(pred, len(db.test)), num_leaves, size)
        if(len(db.x_names) == 0 or size_est < self.noise_est):
            (pred, size) = self.leaf(db, eps)
            return (np.repeat(pred, len(db.test)), 1, size)
        else:
            idx = int(np.random.uniform() * len(db.x_names))
            attr = db.x_names[idx]
            new_x = db.x_names[db.x_names != attr]

            preds = np.repeat(db.train[db.y_name].iloc[0], len(db.test))
            num_leaves = 0
            cnts = pd.Series(0, db.train[db.y_name].cat.categories)
            new_size_est = size_est / len(db.train[attr].cat.categories)
            for att in db.train[attr].cat.categories:
                train_split = db.train[db.train[attr] == att]
                test_split_loc = db.test[attr] == att
                db_new = DPrivacy.Database(
                        train_split, db.test[test_split_loc], new_x, db.y_name)
                p, nl, c = self.decision_helper(db_new, eps, new_size_est)
                num_leaves += nl
                preds[test_split_loc] = p
                cnts += c

            SNR = eps*cnts.sum() / (self.C * np.sqrt(2*num_leaves))
            if(SNR < 1):
                pred = cnts.keys()[cnts.argmax()]
                return (np.repeat(pred, len(db.test)), num_leaves, cnts)
            else:
                return (preds, num_leaves, cnts)
    def get_preds(self):
        return self.decision_helper(self.db, self.budget, len(self.db.train))
    def eval_annotation(self, feat):
        pass

#TODO: Features should be used to help set algorithmic parameters (Big task)
class ChoiceMaker:
    def __init__(self, db, budget, nt, depth):
        (feats, used) = get_features(db, budget)
        budget -= used
        alglist = [FS(db, budget, depth), MA(db, budget, depth), Jag(db, budget, nt)]
        perfs = np.array([x.eval_annotation(feats) for x in alglist])
        self.alg = alglist[perfs.argmax()]
    def get_accuracy(self):
        return self.alg.get_accuracy()

#import cProfile
#ma = MA(dblist['bind'], 5, 5,)
#cProfile.run('ma.get_accuracy()')
