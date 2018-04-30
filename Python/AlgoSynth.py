import DTrees
import DPrivacy
exec(open('GetGraphs.py').read())
  
def gen_data_helper(size, i, attr_probs, p_drop, min_dist):
    if(i == len(attr_probs)-1 or i >= min_dist and p_drop < np.random.uniform()):
        col = pd.DataFrame()
        while(i < len(attr_probs) - 1):
            distr = attr_probs[i]
            col[i] = np.random.choice(range(0, len(distr)), size, p=distr)
            i+=1
        distr = attr_probs[i]
        perm = np.random.choice(range(0, len(distr)), len(distr), False, distr)
        col[i] = np.random.choice(range(0, len(distr)), size, p=perm)
        return col
    p = attr_probs[i]
    num_each = np.random.multinomial(size, p)
    D = pd.DataFrame()
    for j in range(0, len(num_each)):
        sub_d = gen_data_helper(num_each[j], i+1, attr_probs, p_drop, min_dist)
        sub_d[i] = j
        D = D.append(sub_d)
    return D

#Size: number of points.
#Attr_probs[i][j]: probability that attribute i takes value j. Assumed that last point is class
def gen_data(size, attr_probs, p_drop=0.5, min_dist=2):
    for i in range(0, len(attr_probs)):
        p = attr_probs[i]
        if(np.isscalar(p)):
            p = np.ones(p)/p
        attr_probs[i] = p
    return gen_data_helper(size, 0, attr_probs, p_drop, min_dist)

#Domain Size: nat
#Total Domain Size: nat
#Number of Y Classes: nat
#Total Size: nat
#Epsilon: real
#acc: real

def collect_on_trace(db, eps, tot_dom_size, dep, util_func, mx):
    y = db.train[db.y_name]
    cnts = y.value_counts()
    #Computataional Efficiency
    if(util_func.get_ent(cnts) == 0 or dep == mx or len(db.test) == 0 or len(db.train) == 0):
        return pd.DataFrame()
    #acc1: Expected Performance
    pred = y.cat.categories[cnts.idxmax()]
    acc1 = (db.test[db.y_name] == pred).sum()
    #acc2: Expected performance on greedy splitting
    #acc3: Expected performance on splitting into 3
    utils = np.array([util_func.eval(db.train[x], y) for x in db.x_names])
    col = db.x_names[utils.argmax()]
    utils = utils-max(utils)
    weights = np.exp(eps*utils / (2*util_func.sens))
    prob = weights / sum(weights)
    acc2 = 0
    acc3 = 0
    #Compute accuracy of every column
    for i in range(0, len(db.x_names)):
        col_name = db.x_names[i]
        X_train = db.train[col_name]
        Y_train = db.train[db.y_name]
        X_test = db.test[col_name]
        Y_test = db.test[db.y_name]
        a = 0
        for C in X_train.cat.categories:
            most_freq_cat_idx = (Y_train[X_train == C]).value_counts().idxmax()
            most_freq_att = Y_train.cat.categories[most_freq_cat_idx]
            a += (Y_test[X_test == C] == most_freq_att).sum()
        acc2 += a*prob[i]
        acc3 += a
    acc3 /= len(db.x_names)

    #diff is distributed as Laplace(1/eps) centered around acc2 - acc1
    #diff = (acc2 - acc1) / len(db.test)
    szs = [len(db.train[x].cat.categories) for x in db.x_names]
    dom_size = np.array(szs).prod()
    class_size = len(db.train[db.y_name].cat.categories)
    #noisy_cnts = DPrivacy.hist_noiser(cnts, eps)
    #big_frac = noisy_cnts.max() / noisy_cnts.sum()
    #current domain size, total domain size, num y classes, tot_size, epsilon
    row = pd.DataFrame([(dom_size, tot_dom_size, class_size, len(db.train), eps, 
            acc1, acc2, acc3)])
    new_x = db.x_names[db.x_names != col]
    for C in db.train[col].cat.categories:
        train_split = db.train[db.train[col] == C]
        test_split = db.test[db.test[col] == C]
        db_new = DPrivacy.Database(train_split, test_split, new_x, db.y_name)
        row = row.append(collect_on_trace(db_new, eps, tot_dom_size, dep+1, util_func, mx))
    return row

def get_size(db):
    
    szs = [len(db.train[x].cat.categories) for x in db.x_names]
    return np.array(szs).prod()

def collect(db, eps, max_dep = 5):
    C = DPrivacy.ConditionalEntropy(len(db.train))
    max_dep = min(len(db.x_names), max_dep)
    D = collect_on_trace(db, eps, get_size(db), 0, C, max_dep)
    D.columns = ['domsize', 'tot_domsize', 'csize', 'nrow', 'eps', 'acc_diff']
    return D
"""
L1 = [collect(nurs, eps, 5) for eps in [0.2, 0.4, 0.7, 1.0]]
L2 = [collect(bind, eps, 5) for eps in [0.2, 0.4, 0.7, 1.0]]
L3 = [collect(loan, eps, 5) for eps in [0.2, 0.4, 0.7, 1.0]]
L = pd.concat([pd.concat(L1), pd.concat(L2), pd.concat(L3)])

y = np.array(L.diff)
Xs = L.drop('diff', 1)
X_trans = []
for c1 in Xs:
    r = np.array(Xs[c1])
    X_trans.append(r*r)
    X_trans.append(np.sqrt(r))
    X_trans.append(np.log(r))
    for c2 in Xs:
        if(c1 != c2):
            X_trans.append(r / np.array(Xs[c2]))

"""
