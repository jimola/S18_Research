import DTrees
import DPrivacy
import numpy as np
import pandas as pd
import importlib
importlib.reload(DTrees)
importlib.reload(DPrivacy)

from multiprocessing import Pool
import pickle

np.random.seed(1234)
nurs = pd.read_csv('../datasets/nursery.data', header=None)
nurs = DPrivacy.Database.from_dataframe(nurs)
ttt = pd.read_csv('../datasets/tic-tac-toe.data', header=None)
ttt = DPrivacy.Database.from_dataframe(ttt)
bind_raw = pd.read_csv('../datasets/1625Data.txt', header=None)
bind = pd.DataFrame(np.array(list(map(lambda x: bind_raw[0].str.slice(x, x+1),
    np.arange(0, 8)))).T)
bind[9] = bind_raw[1]
bind = DPrivacy.Database.from_dataframe(bind)
contra = pd.read_csv('../datasets/cmc.data', header=None)
#contra = pd.read_csv('../data-unsorted/contra/cmc.data', header=None)
contra[0] = contra[0] / 5
contra = DPrivacy.Database.from_dataframe(contra)
#loan = pd.read_csv('../data-unsorted/loan/student-loan.csv')
loan = pd.read_csv('../datasets/student-loan.csv')
loan = DPrivacy.Database.from_dataframe(loan)
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

#TODO: implement more dbs
dblist = {'nurs': nurs, 'ttt': ttt, 'bind': bind, 'contra': contra, 
        'loan': loan, 'student': student}

alglist = {'FS': DTrees.FS, 
        'MA': DTrees.MA, 'Jag': DTrees.Jag, 'CM': DTrees.ChoiceMaker}

paramlist = {'FS': [(2,),(3,),(5,)],
             'MA': [(2,),(3,),(5,)],
        'Jag': [(3,),(5,),(8,)]}

eps_vals = np.concatenate(([0.5], np.arange(1,10)))

def get_acc(args):
    alg, nm, e, p = args
    return alglist[alg](dblist[nm], e, *p).get_accuracy()

def collect_data(alglist, dblist, eps_vals, paramlist, reps=10):
    try:
        data = pickle.load(open('data.p', 'rb'))
    except:
        data = pd.DataFrame()
        data['alg'] = data['database'] = ''
        data['eps'] = data['perf'] = 0
        data['params'] = ''
    l = []
    for a in alglist:
        for nm in dblist:
            for e in eps_vals:
                for p in paramlist[a]:
                    for i in range(0, reps):
                        l.append((a, nm, e, p))
                        print(a, nm, e, p)
    if(__name__ == '__main__'):
        pool = Pool(processes=10)
        res = pool.map(get_acc, l)
        new_db = pd.DataFrame(l)
        new_db.columns = ['alg', 'database', 'eps', 'params']
        new_db['perf'] = res
        data = data.append(new_db)
        pickle.dump(data, open('data.p', 'wb'))
        return data

def get_annotations(alglist, dblist, eps_vals, paramlist):
    try:
        data = pickle.load(open('annot.p', 'rb'))
    except:
        data = pd.DataFrame()
        data['alg'] = data['database'] = ''
        data['eps'] = data['annotation'] = 0
        data['params'] = ''
    l = []
    for a in alglist:
        for nm in dblist:
            for e in eps_vals:
                for p in paramlist[a]:
                    F = DTrees.get_features(dblist[nm], e)[0]
                    l.append((a, nm, e, p, 
                        alglist[a](dblist[nm], e, *p).eval_annotation(F)))
    new_db = pd.DataFrame(l)
    new_db.columns = ['alg', 'database', 'eps', 'params', 'annotation']
    data = data.append(new_db)
    pickle.dump(data, open('annot.p', 'wb'))
    return data

#Selects a db, select param values, sorts by epsilon, and splits by the algorithms
#DB has: alg, database, eps, params, performance
"""
When we plot the datasets, we can show three dimensions in addition to performance (y-axis):
    x-axis, different plots on the same graph, and different graphs. Since the data
    has 4 dimensions, one has to be ignored.
"""

#We ignore the first element of the permutation by taking the the value specified by
#ignore_value. If None is specified, we take the first one
#TODO: Make sure epsilon is ordered
def graph_data(data, col_perm, y_key = 'perf'):
    D = data.drop(col_perm[0], 1).groupby(col_perm[1:]).mean()
    i = 1
    L = (len(D.index.levels[0]) + 1) // 2
    A = 2
    if(len(D.index.levels[0]) == 1):
        A = 1
    for g in D.index.levels[0]:
        plt.subplot(L, A, i)
        plot = []
        for p in D.index.levels[1]:
            plot.append(np.array(D.loc[g,p][y_key]))
        plt.title(g)
        plt.plot(D.index.levels[2], np.array(plot).T)
        plt.legend(D.index.levels[1])
        i += 1
def split_by_params(db, algdb_to_params):
    ans = pd.DataFrame()
    for a in algdb_to_params:
        for d in algdb_to_params[a]:
            ans = ans.append(db[(db.alg == a) & (db.params == algdb_to_params[a][d]) & 
                (db.database == d)])
    return ans

def do_graphs():
    data = pickle.load(open('data.p', 'rb'))
    data = split_by_params(data, 
            {'FS': {'bind': (5,), 'ttt': (5,), 'nurs': (5,)},
             'MA': {'bind': (3,), 'ttt': (5,), 'nurs': (5,)},
             'Jag': {'bind': (5,), 'ttt': (5,), 'nurs': (5,)}})
    graph_data(data, ['params', 'database', 'alg', 'eps'])


def do_graphs2():
    data = pickle.load(open('data.p', 'rb'))
    data = split_by_params(data, 
            {'FS': {'bind': (5,), 'ttt': (5,), 'nurs': (5,), 'loan': (5,)},
                'MA': {'bind': (3,), 'ttt': (5,), 'nurs': (5,), 'loan': (5,)},
                'Jag': {'bind': (5,), 'ttt': (5,), 'nurs': (5,), 'loan': (5,)},
                'CM': {'bind': (5,5), 'ttt': (5,5), 'nurs': (5,5), 'loan': (5,5)}})
    graph_data(data, ['params', 'database', 'alg', 'eps'])
"""
def select_db_split_algos(dg, dbname, params):
    dg2 = dg[dg.index.get_level_values('database') == dbname]
    mat = []
    for a in alglist:
        mat.append(np.array(dg2['perf'][(dg2.index.get_level_values('alg') == a) &
                                        (dg2.index.get_level_values('params') == params[a])]))
    return np.array(mat).T

grps = data.groupby(['alg', 'database', 'eps'])
dgm = grps.mean()
dgs = grps.agg(np.std)

def add_graph_w_err(dbname, ax=plt.subplot(111)):
    sm = select_db_split_algos(dgm, dbname).T
    ss = select_db_split_algos(dgs, dbname).T
    for i in [0,1,2]:
        ax.errorbar(eps_vals, sm[:, i], yerr=ss[:, i])

#Plot all 6 at once
plt.subplot(321)
plt.title('nurs')
plt.plot(eps_vals, select_db_split_algos(dgm, 'nurs').T)
plt.subplot(322)
plt.title('ttt')
plt.plot(eps_vals, select_db_split_algos(dgm, 'ttt').T)
plt.subplot(323)
plt.title('bind')
plt.plot(eps_vals, select_db_split_algos(dgm, 'bind').T)
plt.subplot(324)
plt.title('contra')
plt.plot(eps_vals, select_db_split_algos(dgm, 'contra').T)
plt.subplot(325)
plt.title('loan')
plt.plot(eps_vals, select_db_split_algos(dgm, 'loan').T)
plt.subplot(326)
plt.title('student')
plt.plot(eps_vals, select_db_split_algos(dgm, 'student').T)
plt.legend(list(alglist.keys()))

params = {'Friedman and Schuster': (5,), 'Mohammed et al.': (5,), 'Jagannathan et al.': (5,), 
        'ChoiceMaker': (5,5)}
"""

