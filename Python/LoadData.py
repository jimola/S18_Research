#Class representing a Database
class Database:
    def __init__(self, train, test, x_names, y_name):
        self.train=train
        self.test=test
        self.y_name=y_name
        self.x_names=x_names
    @classmethod
    def from_dataframe(cls, d, y_idx=-1, cutoff=0.7):
        for x in d.columns:
            d[x] = np.unique(d[x], return_inverse=True)[1]
        d = d.reindex(np.random.permutation(d.index))
        cutoff = int(cutoff*len(d))
        y_name = d.columns[y_idx]
        x_names = d.columns[d.columns != y_name]
        return cls(d[:cutoff], d[cutoff:], x_names, y_name)

nurs = pd.read_csv('../datasets/nursery.data', header=None)
nurs = Database.from_dataframe(nurs)
ttt = pd.read_csv('../datasets/tic-tac-toe.data', header=None)
ttt = Database.from_dataframe(ttt)
bind_raw = pd.read_csv('../datasets/1625Data.txt', header=None)
bind = pd.DataFrame(np.array(list(map(lambda x: bind_raw[0].str.slice(x, x+1),
    np.arange(0, 8)))).T)
bind[9] = bind_raw[1]
bind = Database.from_dataframe(bind)
contra = pd.read_csv('../data-unsorted/contra/cmc.data', header=None)
contra[0] = contra[0] / 5
contra = Database.from_dataframe(contra)
loan = pd.read_csv('../data-unsorted/loan/student-loan.csv')
loan = Database.from_dataframe(loan)
student = pd.read_csv('../data-unsorted/student/student-processed.csv')
student = Database.from_dataframe(student)
#TODO: implement more dbs

dblist = {'nurs': nurs, 'ttt': ttt, 'bind': bind, 'contra': contra, 
        'loan': loan, 'student': student}

