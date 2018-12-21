
#The code we use to preprocess the adult dataset for log_regression_data.
#This file must be run from within LogRegChoice.py
adult = pd.read_csv('data/adult.data', header=None)
adult = adult.rename(columns = {0: 'age', 1: 'workclass', 2: 'fnlwgt', 3: 'education', 
                                4: 'education_num', 5: 'marital_status', 6: 'occupation',
                                7: 'relationship', 8: 'race', 9: 'sex', 10: 'captial_gain',
                                11: 'captial_loss', 12: 'hours-per-week', 13: 'native_country', 14: 'TARGET'})
adult = adult.drop('education', axis=1)

le = LabelEncoder()
adult['TARGET'] = le.fit_transform(adult['TARGET'])

pickle.dump(adult, open('log_regression_data/adult_preprocess.pkl', 'wb'))
