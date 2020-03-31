#The code we use to preprocess the magic dataset for log_regression_data.

spam = pd.read_csv('../data/spambase.csv', header=None)
spam = spam.sample(spam.shape[0], replace=False)
pickle.dump(spam, open('spam_preprocess.pkl', 'wb'))
