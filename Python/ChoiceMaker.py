
class ChoiceMaker:
    """Class of ChoiceMaker object
    
    Parameters
    ----------

    mfs: class computing database metafeatures such as number of rows, domain
    size, and epsilon. Must have an eval method.

    algs : list of algortihms to be selected. Must implement a run and an error
    method.

    mf_eval : Dataframe of evaluated database metafeatures. mf_eval[i][j] is 
    metafeature j evaluated on database i of training set.

    alg_perfs : Dataframe of algorithm performances. alg_perfs[i][j] is the 
    performance of algorithm j on database i of the training set.

    model : Machine Learning model to train with. Must implement a train and
    fit method
     
    """
    def __init__(self, mfs, algs, mf_eval, alg_perfs, model):
        self.model = model
        self.model.fit(mf_eval, alg_perfs)
        self.metafeatures = mfs
        self.algs = dict([(a.name, a) for a in algs])
        self.alg_perfs = alg_perfs
        self.mf_eval = mf_eval

    @classmethod
    def create_regret_based(cls, train_set, alg_list, model, mfs):
        """
        Convenience method for creating a ChoiceMaker

        Parameters
        ----------
        
        train_set : iterable of inputs on which algorithms are run. inputs will
        be a class that usually include a database and must include epsilon as
        members.

        alg_list : list of algorithms to be selected from. Must implement a run
        and an error method.

        model : Machine Learning model for training. Must implement a train and
        a fit method

        mfs : metafeature class. Must implement an eval method.

        """
        Xs = pd.DataFrame()
        y = pd.DataFrame()
        for t in train_set:
        #TODO dynamic test for error vs score method
            errs = pd.DataFrame(dict([(a.name, a.error(t)) for a in alg_list]),
                    index=[0])
            y = y.append(errs, ignore_index=True)
            Xs = Xs.append(pd.DataFrame(mfs.eval(t)), ignore_index=True)
        m = np.min(np.array(y), axis=1)
        regrets = y.divide(m, axis='index')
        return cls(mfs, alg_list, Xs, regrets, model)
    
    """
    @classmethod
    def create_regret_based_new(cls, train_set, alg_list, model, mfs):
        Xs = pd.DataFrame()
        y = pd.DataFrame()
    """
            

    def mkChoice(self, data, ratio=0.2):
        """
        Method for computing Choice.

        Parameters
        ----------
        
        data : input class. Must include epsilon as a member!

        ratio : ratio of epsilon to be used on computing metafeatures vs.
        running the actual algorithm. Default: 0.2

        Returns
        -------

        Best algorithm as selected by model run on data

        """
        eps = data.epsilon
        mf_max_budget = ratio*eps
        data.epsilon = eps-mf_max_budget
        mfs = self.metafeatures.eval(data)
        (best_alg, used) = self.model.predict(mfs, mf_max_budget,
                self.metafeatures.sens)
        data.epsilon = eps-used
        return self.algs[best_alg].run(data)

