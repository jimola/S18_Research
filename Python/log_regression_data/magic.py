#The code we use to preprocess the magic dataset for log_regression_data.

magic = pd.read_csv('../data/magic.csv', header=None)
magic[10] = np.unique(magic[10], return_inverse=True)[1]
pickle.dump(magic, open('log_regression_data/magic_preprocess.pkl', 'wb'))
