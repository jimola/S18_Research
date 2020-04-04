#The code we use to preprocess the default dataset for log_regression_data.

D = pd.read_csv('../data/default.csv')
D = D.rename({'default payment next month': 'TARGET'}, axis=1)

pickle.dump(D, open('default_preprocess.pkl', 'wb'))
