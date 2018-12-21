#The code we use to preprocess the default dataset for log_regression_data.
#This file must be run from within LogRegChoice.py

default = pd.read_csv('data/application_train.csv')
default['TARGET'] = le.fit_transform(default['TARGET'])
#For simplicity, get rid of all columns with missing data
default = default[default.columns[default.notnull().all()]].sample(10000)

new_cols = list(default.columns)
new_cols[1] = new_cols[-1]
new_cols[-1] = 'TARGET'
default = default[new_cols]

pickle.dump(default, open('log_regression_data/default_preprocess.pkl', 'wb'))
