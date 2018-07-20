import pandas as pd
import numpy as np
import DPrivacy as dp
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# TODO
#
# - Replace the metafeature class with a callable object.
# - Perhaps we should rename error to score, which is more general.


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
        # TODO dynamic test for error vs score method
        X = pd.DataFrame([mfs.eval(t) for t in train_set])
        y = pd.DataFrame([dict([(a.name, a.error(t)) for a in alg_list])
                          for t in train_set])
        regrets = y.subtract(np.min(np.array(y), axis = 1), axis = 'index')
        return cls(mfs, alg_list, X, regrets, model)

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

class DTChoice:
    """Choice maker based on sklearn decision trees

    Parameters
    ----------

    train_set: A list of public databases

    mfs: A callable object for computing metafeatures on databases.  The
    returned metafeatures must be a dictionary object mapping metafeature names
    to their values. The mfs object must have a sensitivity attribute, with is a
    dictionary mapping metafeature names to their sensitivities.

    algs: A dictionary mapping names to algorithms. Each algorithm must implement
    a run method, which executes the algorithm on a database, and an error
    method, which computes the algorithm's error on a database.
    
    reps (default=1): Number of times to call the error method in case more
    precision is desired.

    y (optional, default=None): List of algorithms evaluated on train_set
    because it may be expensive to run these algorithms and their results may 
    already be around

    """

    def __init__(self, train_set, mfs, algs, reps=1, y=None):
        self.metafeatures = mfs
        self.algs = algs
        self.X = pd.DataFrame([mfs(t) for t in train_set])
        if y is None:
            self.y = pd.DataFrame([{name: sum([alg.error(t) 
                                   for x in range(0, reps)]) / reps
                                   for name, alg in algs.items()}
                                   for t in train_set])
        else:
            self.y = y

        self.regrets = self.y.subtract(np.min(np.array(self.y), axis = 1),
                                       axis = 'index')
        self.retrain_model()
    
    def update_metas(self, train_set, mfs):
        self.X = pd.DataFrame([mfs(t) for t in train_set])
        self.retrain_model()

    def retrain_model(self):
        self.model = DecisionTreeClassifier()
        self.model = self.model.fit(self.X, self.y.idxmin(axis=1))

    def get_best_alg(self, data, budget):
        sens = self.metafeatures.sensitivities

        X = self.metafeatures(data)
        noisy_X = pd.DataFrame([{name: value + dp.laplacian(budget / len(sens),
                                                            sensitivity = sens[name])
                                 for name, value in self.metafeatures(data).items()}])
        best = self.model.predict(noisy_X)[0]
        return best

    def choose(self, data, ratio = 0.2):
        budget = data.epsilon*ratio
        best = self.get_best_alg(data, budget)
        data.epsilon = data.epsilon - budget
        return self.algs[best].run(data)

    def get_regret(self, data, ratio=0.2, reps=1):
        #data = copy.copy(data)
        budget = data.epsilon*ratio
        ys = pd.DataFrame({name: sum([alg.error(data)
                                for x in range(0, reps)]) / reps
                                for name, alg in self.algs.items()},
                           index=[0])

        best_alg = self.algs[self.get_best_alg(data, budget)]
        data.epsilon = data.epsilon - budget
        R = best_alg.error(data)
        ys['cm'] = R
        return ys - ys.min(axis=1)[0]

    def get_approximate_regret(self, return_std=False, test_ratio=0.3):
        """
        Splits data into training and test and returns average regrets on the
        test split for each algorithm and for this DTChoice object.

        The DTChoice regret is approximate (and an underestimate) for two 
        reasons. Let A = ratio*epsilon and B = (1-ratio)*epsilon.

        First, we don't add Laplace(A) noise to the metafeatures when we 
        predict on them.

        Second, the algorithm we choose isn't run with B budget---it's run with
        epsilon budget instead.

        """
        X_train, X_test, y_train, y_test = train_test_split(self.X,
                self.y, test_size=test_ratio)
        model = DecisionTreeClassifier()
        model.fit(X_train, y_train.idxmin(axis=1))
        algs = model.predict(X_test)
        perfs = y_test.lookup(y_test.index, algs)
        R = np.concatenate((np.array(y_test), perfs[:, None]), axis=1)
        R = R - np.min(R, axis=1)[:, None]
        if(return_std):
            return (R.mean(axis=0), R.std(axis=0))
        else:
            return R.mean(axis=0)
