d = pd.read_csv('../datasets/nursery.data', header=None)

#Class representing a Database
class Database:
    def __init__(self, train, test, x_names, y_name):
        self.train=train
        self.test=test
        self.y_name=y_name
        self.x_names=x_names
    @classmethod
    def from_file(cls, name, y_idx, cutoff=0.7, header=None):
        d = pd.read_csv(name, header=header)
        for x in d.columns:
            d[x] = np.unique(d[x], return_inverse=True)[1]
        d = d.reindex(np.random.permutation(d.index))
        cutoff = int(cutoff*len(d))
        y_name = d.columns[y_idx]
        x_names = d.columns[d.columns != y_name]
        return cls(d[:cutoff], d[cutoff:], x_names, y_name)

class Controller:
    def __init__(self, db, max_depth, budget, nt=1):
        self.max_depth = max_depth
        self.db = db
        self.budget = budget/nt
        self.init_numcols = len(db.x_names)
        self.nt = nt
    def leaf(self, db, eps):
        freq_table = pd.value_counts(db.train[db.y_name])
        noisy_counts = hist_noiser(freq_table, eps)
        return freq_table.keys()[noisy_counts.argmax()]
    def decision_helper(self, db, eps):
        (used, attr) = self.get_action(db)
        eps -= used
        if(attr == None):
            pred = self.leaf(db, eps)
            return np.repeat(pred, len(db.test))
        else:
            new_x = db.x_names[db.x_names != attr]
            preds = np.repeat(db.train[db.y_name].iloc[0], len(db.test))
            for att in iter(np.unique(db.train[attr])):
                train_split = db.train[db.train[attr] == att]
                test_split_loc = db.test[attr] == att
                db_new = Database(train_split, db.test[test_split_loc], new_x, db.y_name)
                preds[test_split_loc] = self.decision_helper(db_new, eps)
            return preds
    def get_preds(self):
        nrow = len(db.test)
        L = np.array(list(map(lambda x: non.decision_helper(db, 5), np.arange(1,10))))
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
    def get_accuracy(self):
        preds = self.decision_helper(self.db, self.budget)
        c = sum(preds == self.db.test[self.db.y_name])
        return c / len(self.db.test)

class NonPrivate(Controller):
    def __init__(self, db, max_depth):
        Controller.__init__(self, db, max_depth, 0)
        self.util_func = ConditionalEntropy(len(db.train))
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        y = db.train[db.y_name]
        if(depth >= self.max_depth or len(y.unique()) <= 1):
            return(0, None)
        utils = list(map(lambda x: self.util_func.eval(db.train[x], y), db.x_names))
        idx = exp_mech(utils, 0, None)
        return (0, db.x_names[idx])

#Friedman and Schuster
class FS(Controller):
    def __init__(self, db, budget, max_depth):
        Controller.__init__(self, db, max_depth, budget)
        self.calc_budget = budget / (2*(max_depth+1))
        col_sizes = db.train.apply(lambda x: len(x.unique()))
        t = max(col_sizes[db.x_names])
        self.stop_const = np.sqrt(2)*t*col_sizes[db.y_name] / self.calc_budget
        self.util_func = ConditionalEntropy(len(db.train))
    def eval_annotation(self):
        pass
    def get_action(self, db):
        nrow = hist_noiser(len(db.train), self.calc_budget)
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth or nrow < self.stop_const):
            return (self.calc_budget, None)
        y = db.train[db.y_name]
        utils = list(map(lambda x: self.util_func.eval(db.train[x], y), db.x_names))
        idx = exp_mech(utils, self.calc_budget, self.util_func.sens)
        return (2*self.calc_budget, db.x_names[idx])

class MA(Controller):
    def __init__(self, db, budget, max_depth):
        Controller.__init__(self, db, max_depth, budget)
        self.calc_budget = budget/(max_depth+1)
        self.util_func = ConditionalEntropy(len(db.train))
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth):
            return (0, None)
        y = db.train[db.y_name]
        utils = list(map(lambda x: self.util_func.eval(db.train[x], y), db.x_names))
        idx = exp_mech(utils, self.calc_budget, self.util_func.sens)
        return (self.calc_budget, db.x_names[idx])

class Jag(Controller):
    def __init__(self, db, budget, nt):
        k = len(db.x_names)
        b = sum(map(lambda x: len(np.unique(db.train[x])), db.x_names))
        b = b / k
        max_dep = min([np.log(len(db.train))/np.log(b)-1, k/2])
        Controller.__init__(self, db, max_dep, budget, nt)
    def get_action(self, db):
        depth = self.init_numcols - len(db.x_names)
        if(depth >= self.max_depth):
            return(0, None)
        idx = int(np.random.uniform() * len(db.x_names))
        return (0, db.x_names[idx])

#Read paper first
class FI(Controller):
    def __init__(self, db, budget, nt):
        Controller.__init__(self, db, 0, budget, nt)
    def get_action(self, db):
        pass
#PyTorch
