from sklearn.preprocessing import KBinsDiscretizer

df = pd.read_csv(open('../data/magic.csv', 'rb'), header=None)
est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='quantile')

cat_df = pd.DataFrame( est.fit_transform(df[df.columns[:10]]), dtype='int' )
cat_df['TARGET'] = df[10]
cat_df = cat_df.apply(lambda c: c.astype('category'))

import pickle
pickle.dump(cat_df, open('magic.pkl', 'wb'))
