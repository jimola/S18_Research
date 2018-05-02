import pickle
def load(name):
    G = pickle.load(open(name, 'rb'))
    df = pd.DataFrame()
    for g in G:
        df = df.append(g[1].groupby(['eps', 'database', 'lnrow', 'ldomsize', 'nclss', 'unif']).mean())
    df.reset_index(inplace=True)
    df.fs5 = 1-df.fs5
    df.ma5 = 1-df.ma5
    df.db1 = 1-df.db1
    df.db3 = 1-df.db3
    df.db7 = 1-df.db7
    return df

def get_best_alg(df):
    best_alg = np.repeat('fs5', len(df))
    best_err = df.fs5.copy()
    mean = (df.fs5 + df.ma5 + df.db1 + df.db3 + df.db7)/5
    for c in ['ma5', 'db1', 'db3', 'db7']:
        err = df[c]
        best_alg[err < best_err] = c
        best_err[err < best_err] = err[err < best_err]

    best_alg_stubbed = best_alg.copy()
    best_alg_stubbed[mean-best_err < 0.03] = 'NA'
    df['mean'] = mean
    df['best_err'] = best_err
    df['best_alg'] = best_alg
    df['best_alg_s'] = best_alg_stubbed
    return df

def get_dsets(idx, sizes = (0.3, 1), seed=12345):
    tr_pct, te_pct = sizes
    def sample_pct(db, pct):
        L = len(db)
        real_test = db.loc[np.random.choice(L, int(pct*L), replace=False)]
        return real_test
    np.random.seed(seed)
    reals = [get_best_alg(load('ttt.p')),
             get_best_alg(load('bind.p')),
             get_best_alg(load('nurs.p')),
             get_best_alg(load('loan.p'))]
    real_test = sample_pct(reals[idx], tr_pct)
    real_train = pd.DataFrame()
    for x in range(0, 4):
        if(x != idx):
            real_train = real_train.append(reals[x])
    real_train.reset_index(drop=True, inplace=True)
    real_train = sample_pct(real_train, tr_pct)
    r1 = get_best_alg(load('rand.p'))
    r2 = get_best_alg(load('rand2.p'))
    r3 = get_best_alg(load('rand3.p'))
    r4 = get_best_alg(load('rand4.p'))
    sim_data = r1.append(r2).append(r3).append(r4).reset_index(drop=True)
    sim_data = sample_pct(sim_data, te_pct)
    cutoff = int(0.75*len(sim_data))
    train = sim_data[:cutoff].append(real_train).reset_index(drop=True)
    test = sim_data[cutoff:].append(real_test).reset_index(drop=True)
    return (train, test)

c_dict = {'ma5': 'green', 'db1': 'orange', 'db3': 'yellow', 
            'db7': 'red', 'fs5': 'blue', 'NA': 'gray'}

def get_scatter(r):
    plt.scatter(r.lnrow, r.eps, c=to_color(r.best_alg, c_dict))
def to_color(r, color_dict):
    for c in color_dict:
        r[r == c] = color_dict[c]
    return r

#get_scatter(t_data)
#get_scatter(b_data)
#get_scatter(n_data)
#get_scatter(l_data)

alglist = ['ma5', 'db1', 'db3', 'db7', 'fs5']
def get_regrets(df):
    dat = df[alglist]
    M = dat.min(axis=1)
    regrets = dat.divide(M, axis=0)
    return regrets.mean(axis=0)

def get_model_err(preds, test):
    err = np.zeros(len(test))
    for a in alglist:
        err[preds == a] = test.loc[preds == a, a]
    err[preds == 'NA'] = test.loc[preds == 'NA', 'mean']
    return err

def get_model_regrets(train, test):

    x_cols = ['eps', 'lnrow', 'ldomsize', 'nclss', 'unif']
    X = train[x_cols]
    Xt = test[x_cols]
    y = train.best_alg_s
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    yp = clf.predict(Xt)
    dat = test[alglist].copy()
    model_err = get_model_err(yp, test)
    dat['model'] = model_err
    M = dat.min(axis=1)
    regrets = dat.divide(M, axis=0)
    return regrets.mean(axis=0)

