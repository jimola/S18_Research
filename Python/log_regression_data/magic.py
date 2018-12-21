#The code we use to preprocess the magic dataset for log_regression_data.
#This file must be run from within LogRegChoice.py

magic = pd.read_csv('data/magic.csv', header=None)
magic[10] = le.fit_transform(magic[10])
pickle.dump(magic, open('log_regression_data/magic_preprocess.pkl', 'wb'))
