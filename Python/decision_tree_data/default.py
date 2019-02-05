def discretize(col, n_quantiles):
    L = np.linspace(0, 1, n_quantiles, endpoint=False) + 1.0/n_quantiles
    L = col.quantile(L)
    return L.searchsorted(col)

default = pd.read_csv('data/default.csv', header=1)
default = default.drop('ID', axis=1)

cols = [c for c in default.columns if 'AMT' in c]
default[cols] = default[cols].apply(lambda x: discretize(x, 5))
#Columns remaining: AGE, LIMIT_BAL, PAY_n,
default.LIMIT_BAL = discretize(default.LIMIT_BAL, 10)
default.AGE = discretize(default.AGE, 6)
default = default.apply(lambda x: x.astype('category'))

pickle.dump(default, open('decision_tree_data/default.pkl', 'wb'))

