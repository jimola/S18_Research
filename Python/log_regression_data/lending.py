#The code we use to preprocess the default dataset for log_regression_data.
#This file must be run from within LogRegChoice.py


lending = pd.read_csv('data/fam_credit_ss.csv')

#Swap first and last column
L = list(lending.columns)
L[0] = L[-1]
L[-1] = 'credit_card'
lending = lending[L]

#Some columns should be changed to object
lending.select_dtypes(['int64']).apply(pd.Series.nunique)

unordered_cols = ['social_security', 'stud_loan', 'medical_exp',
        'marital_status', 'housing', 'health_status']
lending[unordered_cols] = lending[unordered_cols].astype('object')


pickle.dump(lending, open('log_regression_data/lending_preprocess.pkl', 'wb'))
