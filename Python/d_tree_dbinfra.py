class null_cm:
    def __init__(self, depth):
        self.leaf = Leaf()
        self.split = Split()
        self.depth = depth
    def choose(self, db):
        if db.depth < min(db.max_depth, self.depth):
            return self.split.run(db)
        else:
            return self.leaf.run(db)

"""
Performs dataset surgery on a seed_db. Parameters such as target number of rows are fixed here, 
which makes sense because seed databases are roughly the same size in this experiment.

Parameters:
seed_db: dataset to slice up
eps: value of epsilon for the experiments
prng: random number generator
returns a tuple: (regret of algorithm on each db, metafeatures associated with db, and the db itself)
"""
def get_train_dbs(seed_db, eps, max_depth, start_size, branch_factor=2,
        max_size=5000, throwaway=0.7, prng=None):
    if prng is None:
        prng = np.random.RandomState(12345)
    D = []
    for l in range(1, max_depth): 
        for x in range(start_size * branch_factor ** l):
            cols = prng.permutation(seed_db.columns[:-1])
            db_groups = seed_db.groupby(list(cols[:l])).groups
            idxs = db_groups[list(db_groups)[prng.randint(len(db_groups))]]
            L = min(max_size, idxs.size)
            L = prng.randint(throwaway*L, L)
            idxs = prng.choice(idxs, L)
            data = DB(seed_db.loc[idxs, cols[l:]], seed_db.loc[idxs, \
                seed_db.columns[-1]], None, None, epsilon=eps, depth=l)
            D.append(data)
    #Large DBs    
    for x in range(start_size):
        cols = seed_db.columns[:-1]
        L = min(max_size, len(seed_db))
        L = prng.randint(throwaway*L, L)
        new_db = seed_db.sample(L, random_state=prng)
        data = DB(new_db.loc[:, cols], new_db.loc[:, seed_db.columns[-1]], \
                None, None, epsilon=eps, depth=0)
        D.append(data)
    return D

"""
Does a similar thing as get_train_dbs, but makes fewer slices, and the slices are large
because these are databases we want to test on (and we probably won't be using tiny dbs in real life).
"""

def get_test_dbs(seed_db, eps, max_depth, max_size=5000, throwaway=0.7, 
        validation_size=0.7, prng=None):
    if prng is None:
        prng = np.random.RandomState(12345)
    cols = seed_db.columns[:-1]
    y_col = seed_db.columns[-1]
    L = min(max_size / validation_size, len(seed_db))
    L = prng.randint(throwaway*L, L)
    new_db = seed_db.sample(L, random_state=prng).reset_index(drop=True)
    max_depth = min(len(cols), max_depth)
    split = int(validation_size*L)
    d = DB(new_db.loc[:split, cols], new_db.loc[:split, y_col], \
           new_db.loc[split:, cols], new_db.loc[split:, y_col], epsilon=eps,
           max_depth=max_depth)
    return d


