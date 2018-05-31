
class ChoiceMaker:
    """Class of ChoiceMaker object
    
    Parameters
    ----------

    mfs: class computing database metafeatures such as number of rows, domain
    size, and epsilon. Must have an eval method.

    algs: list of algortihms in our selection. Must implement a run and an error method

    mf_eval : Dataframe of evaluated database metafeatures. mf_eval[i][j] is 
    metafeature j evaluated on database i of training set.

    alg_perfs : Dataframe of algorithm performances. alg_perfs[i][j] is the 
    performance of algorithm j on database i of the training set.

    model : Machine Learning model to train with. Must implement a train and
    predict method
     
    """
    def __init__(self, mfs, algs, mf_eval, alg_perfs, model):
        self.model = model
        self.model.fit(mf_eval, alg_perfs)
        self.metafeatures = mfs
        self.algs = dict([(a.name, a) for a in alg_list])
    @classmethod
    def create_regret_based(cls, train_set, alg_list, model, mfs):
        Xs = pd.DataFrame()
        y = pd.DataFrame()
        for t in train_set:
            errs = pd.DataFrame(dict([(a.name, a.error(t)) for a in alg_list]),
                    index=[0])
            y = y.append(errs, ignore_index=True)
            Xs = Xs.append(pd.DataFrame(mfs.eval(t)), ignore_index=True)
        m = np.min(np.array(y), axis=1)
        regrets = y.divide(m, axis='index')
        return cls(mfs, alg_list, Xs, regrets, model)

    def mkChoice(self, db):
        mfs = self.metafeatures.eval(db)
        best_alg = self.model.predict(mfs)
        return self.algs[best_alg].run(db)
