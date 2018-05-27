exec(open('../LoadData.py').read())
import dpcomp_core.algorithm as algorithm
#Left out: HTree
#Use of Q documented below:
OneDimAlgos = [algorithm.ahp.ahp_engine(), #None
               algorithm.dawa.dawa_engine(),
               algorithm.DPcube1D.DPcube1D_engine(),
               algorithm.HB.HB_engine(),
               algorithm.identity.identity_engine(),
               algorithm.mwemND.mwemND_simple_engine(),
              ]

TwoDimAlgos = [algorithm.AG.AG_engine(),
               algorithm.ahp.ahpND_engine(), #None
               algorithm.dawa.dawa2D_engine(),
               algorithm.DPcube.DPcube_engine(),
               algorithm.HB2D.HB2D_engine(),
               algorithm.identity.identity_engine(),
              ]
class NaiveBayesChooser:
    def __init__(alglist, mfeatures, train_dbs, scorefunc):
        pass
    def choose(DB):
        pass
        
    def fitNaiveBayes(DB, epsilon):
        pass
