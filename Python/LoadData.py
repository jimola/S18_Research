import os, sys
direct='../'
sys.path.append(os.path.abspath(direct))
import DPrivacy
import numpy as np
import pandas as pd

np.random.seed(1234)
nurs = pd.read_csv(direct+'../datasets/nursery.data', header=None)
nurs = DPrivacy.Database.from_dataframe(nurs)
ttt = pd.read_csv(direct+'../datasets/tic-tac-toe.data', header=None)
ttt = DPrivacy.Database.from_dataframe(ttt)
bind_raw = pd.read_csv(direct+'../datasets/1625Data.txt', header=None)
bind = pd.DataFrame(np.array(list(map(lambda x: bind_raw[0].str.slice(x, x+1),
        np.arange(0, 8)))).T)
bind[9] = bind_raw[1]
bind = DPrivacy.Database.from_dataframe(bind)
contra = pd.read_csv(direct+'../datasets/cmc.data', header=None)
contra[0] = contra[0] / 5
contra = DPrivacy.Database.from_dataframe(contra)
loan = pd.read_csv(direct+'../datasets/student-loan.csv')
loan = DPrivacy.Database.from_dataframe(loan)
student = pd.read_csv(direct+'../datasets/student-processed.csv')
student = DPrivacy.Database.from_dataframe(student)
votes = pd.read_csv(direct+'../datasets/house-votes-84.data', header=None)
votes = DPrivacy.Database.from_dataframe(votes)

adult = pd.read_csv(direct+'../datasets/adult/adult.data', header=None)
adult = DPrivacy.Database.from_dataframe(adult, cutoff=1)
