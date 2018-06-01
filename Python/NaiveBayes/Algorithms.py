from dpcomp_core.algorithm import *
from dpcomp_core.algorithm.dawa import l1partition
from dpcomp_core.query_nd_union import ndRangeUnion
from dpcomp_core import workload
from dpcomp_core import experiment
from dpcomp_core import metric
from dpcomp_core import dataset

class Data:
    def __init__(self, data, dom_size, scale, wktype, eps):
        self.x = data
        self.dom_size = dom_size
        self.scale = scale
        if(wktype):
            self.Q = workload.Prefix1D(domain_shape_int=dom_size)
        else:
            self.Q = workload.Identity((dom_size,))
        self.wkload_type = wktype
        self.epsilon = eps
    @classmethod
    def from_name(cls, dbname, dom_size, scale, wktype, eps):
        x = dataset.DatasetSampledFromFile(nickname=dbname,
            sample_to_scale=int(scale), reduce_to_dom_shape=int(dom_size),
            seed=12345)
        return cls(x, dom_size, scale, wktype, eps)

def seeder(seed=12345):
    prng = np.random.RandomState(seed)
    while(True):
        yield prng.randint(0, 2**32)

class HistMetafeats:
    def __init__(self):
        self.S = seeder()
    @staticmethod
    def dev(H):
        return abs(H-H.mean()).sum()
    def min_cost(self, H, epsilon):
        P = l1partition.L1partition(H, epsilon, gethist=True, seed=next(self.S))
        cost = sum([HistMetafeats.dev(H[l:r+1]) for l, r in P]) + len(P)/epsilon
        return cost
    def eval(self, data):
        if(isinstance(data.x, dataset.DatasetSampledFromFile)):
            x = data.x.dist * data.x.scale
        else:
            x = data.x
        H = np.around(x).astype('int')
        nnz = (H == 0).sum()
        tvd = HistMetafeats.dev(H)
        cost = self.min_cost(H, data.epsilon)
        return pd.DataFrame({'scale': data.scale, 'shape': data.dom_size,
                             'nnz': nnz, 'tvd': tvd, 'cost': cost}, index=[0])
class HistAlgo:
    def __init__(self, name):
        self.S = seeder()
        self.name = name

    def error(self, data):
        M = metric.SampleError(experiment.Single(data.x, data.Q, self._alg,
            data.epsilon, next(self.S)))
        err = M.compute().error_payload
        return err['TypeI.L2']

    def run(self, data):
        if(isinstance(data.x, dataset.DatasetSampledFromFile)):
            x = np.around(t.x.dist*t.x.scale)
        else:
            x = np.array(data.x)
        R = self._alg.Run(data.Q, x, data.epsilon, next(self.S))
        return R

class Dawa(HistAlgo):
    def __init__(self):
        self._alg = dawa.dawa_engine()
        super().__init__("DAWA")

class Identity(HistAlgo):
    def __init__(self):
        self._alg = identity.identity_engine()
        super().__init__("Id")

class Hb(HistAlgo):
    def __init__(self):
        self._alg = HB.HB_engine()
        super().__init__("Hb")

from pandas.api.types import CategoricalDtype
class DTreeNode:
    def __init__(self, md, qf):
        self.max_depth = md
        self.quality_func = qf
    def leaf(self, y):
        bests = y.columns[np.argmax(np.array(y), axis=1)]
        freq_table = pd.value_counts(bests)
        self.pred = freq_table.idxmax()
        return self
    def train(self, X, y):
        if(len(X.columns) == 0 or self.max_depth == 0):
            return self.leaf(y)
        cur_score = self.quality_func(y)
        best_col = ''
        q_min = cur_score
        for col in X:
            x = X[col]
            if(isinstance(x.dtype, CategoricalDtype)):
                sizes = np.array([(x==c).sum() for c in x.cat.categories]) / len(x)
                scores = np.array([self.quality_func(y[x==c]) for c in x.cat.categories])
                qs = (sizes*scores).sum()
                if(qs < q_min):
                    q_min = qs
                    best_col = (col, )
            else:
                for elem in np.random.choice(x, min(len(x), 50), replace=False):
                    less = y[x < elem]
                    geq = y[x >= elem]
                    qs = (len(less)*self.quality_func(less) +
                            len(geq)*self.quality_func(geq)) / len(x)
                    if(qs < q_min):
                        q_min = qs
                        best_col = (col, elem)
        if(q_min >= cur_score):
            return self.leaf(y)
        self.best_col = best_col
        x = X[best_col[0]]
        if(len(best_col) == 1):
            self.children = dict([(c, DTreeNode(self.max_depth-1, 
                self.quality_func).train(X[x==c], y[x==c]))
                             for c in x.cat.categories])
        else:
            e = best_col[1]
            self.children = [
                    DTreeNode(self.max_depth-1, self.quality_func).train(X[x<e], y[x<e]),
                    DTreeNode(self.max_depth-1, self.quality_func).train(X[x>=e], y[x>=e])]
        return self

    def get_pred(self, x):
        if(hasattr(self, 'pred')):
            return self.pred
        col = self.best_col[0]
        if(len(self.best_col) == 1):
            return self.children[x[col]].get_pred(x)
        split = self.best_col[1]
        if(x.loc[0, col] < split):
            return self.children[0].get_pred(x)
        else:
            return self.children[1].get_pred(x)
class DTree:
    def __init__(self, max_depth, qf):
        self.max_depth = max_depth
        self.quality_func = qf
        
    def fit(self, X, y):
        md = min(self.max_depth, X.shape[1])
        self.dtree = DTreeNode(md, self.quality_func).train(X, y)
        return
    def predict(self, x):
        if(len(x) == 1):
            return self.dtree.get_pred(x)
        else:
            preds = []
            for i in x.index:
                row = x.loc[i, :]
                preds.append(self.dtree.get_pred(row))
            return np.array(preds)
    def score(self, X, y):
        preds = np.array(self.predict(X))
        return (preds == y).sum()

def gini_cnts(cnts):
    probs = cnts / cnts.sum()
    return 1-(probs*probs).sum()
def gini(col):
    cnts = pd.value_counts(col)
    return gini_cnts(cnts)

def group_gini(regrets, theta=1.0):
    y = regrets.columns[np.argmax(np.array(regrets), axis=1)]
    mean_regrets = regrets.mean(axis='index')
    mean_regrets.sort_values(inplace=True)
    last_idx = mean_regrets.index[0]
    num_in_group = 0
    cnts = []
    for i in mean_regrets.index:
        if(mean_regrets[i] - mean_regrets[last_idx] > theta):
            last_idx = i
            cnts.append(num_in_group)
            num_in_group = (y == i).sum()
        else:
            num_in_group += (y == i).sum()
    if(num_in_group  > 0):
        cnts.append(num_in_group)
    cnts = np.array(cnts)
    return gini_cnts(cnts)

S = seeder()
def iter_train_set():
    for name in ['ADULT', 'HEPTH', 'INCOME', 'MEDCOST', 'NETTRACE', 'SEARCHLOGS',
            'PATENT']:
        for dom_size in 2**np.arange(7, 13):
            for scale in 2**np.arange(5, 25):
                yield Data.from_name(name, dom_size, scale, 0, 1.0)

train_set = iter_train_set()
#train_set = [next(T) for i in range(0, 3)]

#cm = ChoiceMaker.create_regret_based(t_set, alg_list, model, mfs)
