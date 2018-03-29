import DTrees
import DPrivacy
import numpy as np

from multiprocessing import Pool
import pickle

np.random.seed(123456)
nurs = pd.read_csv('../datasets/nursery.data', header=None)
nurs = DPrivacy.Database.from_dataframe(nurs)
ttt = pd.read_csv('../datasets/tic-tac-toe.data', header=None)
ttt = DPrivacy.Database.from_dataframe(ttt)
bind_raw = pd.read_csv('../datasets/1625Data.txt', header=None)
bind = pd.DataFrame(np.array(list(map(lambda x: bind_raw[0].str.slice(x, x+1),
    np.arange(0, 8)))).T)
bind[9] = bind_raw[1]
bind = DPrivacy.Database.from_dataframe(bind)
contra = pd.read_csv('../data-unsorted/contra/cmc.data', header=None)
contra[0] = contra[0] / 5
contra = DPrivacy.Database.from_dataframe(contra)
loan = pd.read_csv('../data-unsorted/loan/student-loan.csv')
loan = DPrivacy.Database.from_dataframe(loan)
student = pd.read_csv('../data-unsorted/student/student-processed.csv')
student = DPrivacy.Database.from_dataframe(student)
votes = pd.read_csv('../datasets/house-votes-84.data', header=None)
votes = DPrivacy.Database.from_dataframe(votes)
german = pd.read_csv('../data-unsorted/german/german.data', sep=' ', header=None)
german[1] = (german[1] / 6).astype('int')
german[4] = (german[4] / 100).astype('int')
german[12] = (german[12] / 7).astype('int')
german = DPrivacy.Database.from_dataframe(german, 0)
#TODO: implement more dbs

dblist = {'nurs': nurs, 'ttt': ttt, 'bind': bind, 'contra': contra, 
        'loan': loan, 'student': student}

alglist = {'Friedman and Schuster': DTrees.FS, 
        'Mohammed et al.': DTrees.MA, 'Jagannathan et al.': DTrees.Jag}


eps_vals = np.concatenate(([0.5], np.arange(1,10)))
l = []
for a in alglist:
    for nm in dblist:
        for e in eps_vals:
            for i in range(0,10):
                l.append((a, nm, e, i))
def get_data(args):
    alg, nm, e, i = args
    return alglist[alg](dblist[nm], e, 5).get_accuracy()

try:
    res = pickle.load(open('res.p', 'rb'))
except:
    if(__name__ == '__main__'):
        pool = Pool(processes=10)
        res = pool.map(get_data, l)
        pickle.dump(res, open('res.p', 'wb'))

data = pd.DataFrame(l)
data.columns = ['alg', 'database', 'eps', 'iter']
data['perf'] = res

grps = data.groupby(['alg', 'database', 'eps'])
dgm = grps.mean()
dgs = grps.agg(np.std)
def collapse_groups(dg, dbname):
    dg2 = dg[dg.index.get_level_values('database') == dbname]
    mat = []
    for a in alglist:
        mat.append(np.array(dg2['perf'][dg2.index.get_level_values('alg') == a]))
    mat = np.array(mat)
    return mat

def add_graph_w_err(dbname, ax=plt.subplot(111)):
    sm = collapse_groups(dgm, dbname).T
    ss = collapse_groups(dgs, dbname).T
    for i in [0,1,2]:
        ax.errorbar(eps_vals, sm[:, i], yerr=ss[:, i])

#Plot all 6 at once
plt.subplot(321)
plt.title('nurs')
plt.plot(eps_vals, collapse_groups(dgm, 'nurs').T)
plt.subplot(322)
plt.title('ttt')
plt.plot(eps_vals, collapse_groups(dgm, 'ttt').T)
plt.subplot(323)
plt.title('bind')
plt.plot(eps_vals, collapse_groups(dgm, 'bind').T)
plt.subplot(324)
plt.title('contra')
plt.plot(eps_vals, collapse_groups(dgm, 'contra').T)
plt.subplot(325)
plt.title('loan')
plt.plot(eps_vals, collapse_groups(dgm, 'loan').T)
plt.subplot(326)
plt.title('student')
plt.plot(eps_vals, collapse_groups(dgm, 'student').T)
plt.legend(list(alglist.keys()))

