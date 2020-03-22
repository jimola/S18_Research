
def get_dp_logistic_regression(X_train, y_train, X_test, y_test, 
                             epsilon, ratio, param_list):
    train_eps = epsilon*ratio
    validation_eps = epsilon-train_eps
    #Begin validation
    utils = []
    for C in param_list:
        model = DPLogisticRegression(train_eps, C=C, K=1.02, fit_intercept=True)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = (preds == y_test).sum() / y_test.size
        beta = max(1.0/y_test.size, 1.0/y_train.size)
        score += np.random.exponential(1/validation_eps)*beta*2
        utils.append(score)
    return np.array(utils).argmax()

def test_db_MRE(data_test, ratio=0.8, splits=5, do_private=True):
    kf = model_selection.KFold(splits)
    avgs = []
    for db in data_test:
        avg = 0
        for train_idx, test_idx in kf.split(db.X):
            X_val = db.X.iloc[test_idx]
            y_val = db.y.iloc[test_idx]
            X_train = db.X.iloc[train_idx]
            y_train = db.y.iloc[train_idx]
            X_ttrain, X_ttest, y_ttrain, y_ttest = model_selection.train_test_split(X_train, y_train, test_size=0.2)
            if pd.Series.nunique(y_ttrain) == 1:
                avg += 1.0
                continue
            if do_private:
                eps = db.epsilon
            else:
                eps=100
            idx = get_dp_logistic_regression(X_ttrain, y_ttrain, X_ttest, y_ttest, eps, ratio, C_list)
            alg = DP(C_list[idx])
            if do_private:
                alg.model.set_epsilon(eps - (1-ratio)*eps)
            else:
                alg.model.set_epsilon(100)
            alg.model.fit(X_ttrain, y_ttrain)
            y_hat = alg.model.predict(X_val)
            avg += (y_hat == y_val).sum() / y_val.size
        avg /= splits
        avgs.append(1.0-avg)
    return np.array(avgs)

"""Performance of ERM on each dataset slice. We generate the test sets freshly because
the choicemaker alters epsilon yet we need the same amount of budget.
"""
(adult_data_test, default_data_test, lending_data_test, magic_data_test) = get_test_set()
adult_perf = test_db_MRE(adult_data_test)
lending_perf = test_db_MRE(lending_data_test)
default_perf = test_db_MRE(default_data_test)
magic_perf = test_db_MRE(magic_data_test)

"""Performance of non-private cross-validation on each dataset partition. """
(adult_data_test, default_data_test, lending_data_test, magic_data_test) = get_test_set()
adult_perf_non = test_db_MRE(adult_data_test, do_private=False)
lending_perf_non = test_db_MRE(lending_data_test, do_private=False)
default_perf_non = test_db_MRE(default_data_test, do_private=False)
magic_perf_non = test_db_MRE(magic_data_test, do_private=False)


