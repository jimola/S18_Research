def discretize(col, n_quantiles):
    L = np.linspace(0, 1, n_quantiles, endpoint=False) + 1.0/n_quantiles
    L = col.quantile(L)
    return L.searchsorted(col)


lending = pd.read_csv('../data/fam_credit_ss.csv')

lending.income = discretize(lending.income, 8)
lending.age = discretize(lending.age, 5)
lending.work_hours = discretize(lending.work_hours, 5)
lending.employment = discretize(lending.employment, 6)
lending.auto_insurance = discretize(lending.auto_insurance, 10)

lending = lending.apply(lambda x: x.astype('category'))

pickle.dump(lending, open('lending.pkl', 'wb'))
