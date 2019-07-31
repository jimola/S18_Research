def discretize(col, n_quantiles):
    L = np.linspace(0, 1, n_quantiles, endpoint=False) + 1.0/n_quantiles
    L = col.quantile(L)
    return L.searchsorted(col)

adult = pd.read_csv('../data/adult.data')
adult = adult.rename(columns = {0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education', 
                                4: 'education_num', 5: 'marital_status', 6: 'occupation',
                                7: 'relationship', 8: 'race', 9: 'sex', 10: 'capital_gain',
                                11: 'capital_loss', 12: 'hours_per_week', 13: 'native_country',
                                14: 'TARGET'})

adult.age = discretize(adult.age, 7)
adult.fnlwgt = discretize(adult.fnlwgt, 10)
adult.capital_gain = discretize(adult.capital_gain, 6)
adult.capital_loss = discretize(adult.capital_loss, 6)
adult.hours_per_week = discretize(adult.hours_per_week, 6)

def world_region(elem):
    elem = elem.strip()
    if elem in ['Cuba', 'Jamaica', 'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'Columbia', 'Ecuador', 'Haiti',\
                'Dominican-Republic', 'El-Salvador', 'Guatemala', 'Peru', 'Trinadad&Tobago', 'Nicaragua']:
        return 1
    if elem in ['Philippines', 'Cambodia', 'Thailand', 'Laos', 'Taiwan', 'China', 'Japan',\
                'Outlying-US(Guam-USVI-etc)', 'Hong', 'Vietnam', 'India', 'Iran']:
        return 2
    if elem in ['England', 'Canada', 'France', 'Germany', 'Italy', 'Poland', 'Portugal',\
                'Yugoslavia', 'Scotland', 'Greece', 'Ireland', 'Hungary', 'Holand-Netherlands']:
        return 3
    if elem == 'United-States':
        return 4
    if elem == '?':
        return 5
    assert False
adult.native_country = adult.native_country.apply(world_region)

adult = adult.apply(lambda x: x.astype('category'))

pickle.dump(adult, open('adult.pkl', 'wb'))
