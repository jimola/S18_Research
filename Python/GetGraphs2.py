import numpy as np
import pandas as pd

import DTrees
import DPrivacy
import importlib
importlib.reload(DTrees)
importlib.reload(DPrivacy)

from multiprocessing import Pool
import pickle

def rand_subsets(db, db_name):
    D = dict()
    for x in [0.2, 0.4, 0.6, 0.8, 1]:
        for y in [0.25, 0.50, 0.75, 1]:
            sample_size = int(len(db) * x)
            num_cols = int((len(db.columns)-1) * y)
            for i in range(0, 3):
                cols = np.concatenate(( 
                        np.random.choice(db.columns[0:-1], num_cols, replace=False), 
                        [db.columns[-1]]))
                rows = np.random.choice(db.index.values, sample_size, replace=False)
                D[(db_name, x, y, i)] = DPrivacy.Database.from_dataframe(db[cols].loc[rows])
    return D

np.random.seed(1234)
nurs = pd.read_csv('../datasets/nursery.data', header=None)
#nurs = DPrivacy.Database.from_dataframe(nurs)
nurs = rand_subsets(nurs, 'nurs')
ttt = pd.read_csv('../datasets/tic-tac-toe.data', header=None)
#ttt = DPrivacy.Database.from_dataframe(ttt)
ttt = rand_subsets(ttt, 'ttt')
bind_raw = pd.read_csv('../datasets/1625Data.txt', header=None)
bind = pd.DataFrame(np.array(list(map(lambda x: bind_raw[0].str.slice(x, x+1),
    np.arange(0, 8)))).T)
bind[8] = bind_raw[1]
#bind = DPrivacy.Database.from_dataframe(bind)
bind = rand_subsets(bind, 'bind')
contra = pd.read_csv('../datasets/cmc.data', header=None)
contra[0] = contra[0] / 5
contra = DPrivacy.Database.from_dataframe(contra)
loan = pd.read_csv('../datasets/student-loan.csv')
#loan = DPrivacy.Database.from_dataframe(loan)
loan = rand_subsets(loan, 'loan')
#student = pd.read_csv('../data-unsorted/student/student-processed.csv')
student = pd.read_csv('../datasets/student-processed.csv')
student = DPrivacy.Database.from_dataframe(student)
votes = pd.read_csv('../datasets/house-votes-84.data', header=None)
votes = DPrivacy.Database.from_dataframe(votes)
#german = pd.read_csv('../data-unsorted/german/german.data', sep=' ', header=None)
german = pd.read_csv('../datasets/german.data', sep=' ', header=None)
german[1] = (german[1] / 6).astype('int')
german[4] = (german[4] / 100).astype('int')
german[12] = (german[12] / 7).astype('int')
german = DPrivacy.Database.from_dataframe(german, 0)

def get_acc2(db, eps):
    L = [
        DTrees.FS(db, eps, 5).get_accuracy(),
        DTrees.MA(db, eps, 5).get_accuracy(),
        DTrees.Jag(db, eps, 1).get_accuracy(),
        DTrees.Jag(db, eps, 3).get_accuracy(),
        DTrees.Jag(db, eps, 7).get_accuracy(),
        eps
    ]
    return L
    
def get_gini(sample, num_zeros=0):
    tot_size = num_zeros + len(sample)
    B = np.sort(sample).cumsum().sum() / sample.sum()
    A = (tot_size+1)/2
    return (A-B) / A
def get_size_gini(db, slice_size=4):
    cols = list(np.random.choice(db.x_names, slice_size, False))
    attr_sizes = [len(db.train[x].unique()) for x in db.x_names]
    prods = np.array(attr_sizes).prod()
    sizes = np.array(db.train.groupby(cols)[db.y_name].count())
    num_zeros = prods - len(sizes)
    skew = sizes / len(db.train)
    return get_gini(skew, num_zeros)

def collect_data2(db, db_name, epsvals, reps=10):
    C = len(db.train[db.y_name].cat.categories)
    nrow = len(db.train)
    szs = [len(db.train[x].cat.categories) for x in db.x_names]
    log_dom_size = np.log(szs).sum()
    log_nrow = np.log(nrow)
    res = []
    for r in range(0, reps):
        for e in epsvals:
            res.append(get_acc2(db, e))
    res = pd.DataFrame(res)
    res.columns = ['fs5', 'ma5', 'db1', 'db3', 'db7', 'eps']
    res['database'] = db_name
    res['lnrow'] = log_nrow
    res['ldomsize'] = log_dom_size
    res['nclss'] = C
    res['unif'] = get_size_gini(db, min(len(db.x_names), 4))
    #pickle.dump(data, open('data2.p', 'wb'))
    return res
    
def collect_on_splits(db, db_name, epsvals, reps):
    i = np.random.randint(1, 3)
    cols = np.random.choice(db.x_names, i, replace=False)
    x_new = []
    for x in db.x_names:
        if(not x in cols):
	        x_new.append(x)
    x_new = np.array(x_new)
    vals = [np.random.choice(db.train[x].unique(), 1)[0] for x in cols]
    mask = db.train[cols[0]] == vals[0]
    mask2 = db.test[cols[0]] == vals[0]
    if(len(cols) > 1):
        mask &= db.train[cols[1]] == vals[1]
        mask2 &= db.test[cols[1]] == vals[1]

    db_new = DPrivacy.Database(db.train[mask], db.test[mask2], x_new, db.y_name)
    R = collect_data2(db_new, db_name, epsvals, reps)
    R['slice'] = str(cols) + str(vals)
    return R

def f(params):
    return collect_on_splits(*params)
eps_vals = np.array([0.5] + range(1, 11))
def do_nurs(k):
    return (k, collect_data2(nurs[k], 'nurs', eps_vals, 10))
def do_bind(k):
    return (k, collect_data2(bind[k], 'bind', eps_vals, 10))
def do_loan(k):
    return (k, collect_data2(loan[k], 'loan', eps_vals, 10))
def do_ttt(k):
    return (k, collect_data2(ttt[k], 'ttt', eps_vals, 10))
#nurs2 = DPrivacy.Database(nurs.train.iloc[0:10], nurs.test, nurs.x_names, nurs.y_name)
#bind2 = DPrivacy.Database(bind.train.iloc[0:10], bind.test, bind.x_names, bind.y_name)
eps_vals = np.concatenate(([0.5], np.arange(1, 10)))

if(__name__ == '__main__'):
    pool = Pool(processes=10)
    pickle.dump(pool.map(do_bind, bind), open('bind.p', 'wb'))
    pickle.dump(pool.map(do_nurs, nurs), open('nurs.p', 'wb'))
    pickle.dump(pool.map(do_ttt, ttt), open('ttt.p', 'wb'))
    pickle.dump(pool.map(do_loan, loan), open('loan.p', 'wb'))

