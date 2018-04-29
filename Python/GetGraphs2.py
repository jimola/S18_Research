

dblist = {'nurs': nurs, 'ttt': ttt, 'bind': bind, 'loan': loan}
alglist = {'FS': DTrees.FS, 'MA': DTrees.MA, 'Jag': DTrees.Jag}

def get_acc2(params):
    db, eps = params
    C = len(db.train[db.y_name].cat.categories)
    nrow = len(db.train)
    szs = [len(db.train[x].cat.categories) for x in db.x_names]
    log_dom_size = np.log(szs).sum()
    log_nrow = np.log(nrow)
    D = [C, log_dom_size, log_nrow, eps]
    L = [
        DTrees.FS(db, eps, 2).get_accuracy(),
        DTrees.FS(db, eps, 3).get_accuracy(),
        DTrees.FS(db, eps, 5).get_accuracy(),
        DTrees.MA(db, eps, 2).get_accuracy(),
        DTrees.MA(db, eps, 3).get_accuracy(),
        DTrees.MA(db, eps, 5).get_accuracy(),
        DTrees.Jag(db, eps, 3).get_accuracy(),
        DTrees.Jag(db, eps, 6).get_accuracy(),
        DTrees.Jag(db, eps, 9).get_accuracy()
    ]
    return D+L
    
def collect_data2(db, db_name, epsvals, reps=10):
    try:
        data = pickle.load(open('data2.p', 'rb'))
    except:
        data = pd.DataFrame()
    l = []
    for r in range(0, reps):
        for e in epsvals:
            l.append((db, e))
    if(__name__ == '__main__'):
        pool = Pool(processes=10)
        res = pd.DataFrame(pool.map(get_acc2, l))
        res['database'] = db_name
    data = data.append(res)
    pickle.dump(data, open('data2.p', 'wb'))
    return data
