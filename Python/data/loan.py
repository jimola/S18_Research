import os
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(path, 'student-loan.csv'), header = None)
