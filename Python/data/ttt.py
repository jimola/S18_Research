import os
import pandas as pd

path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(os.path.join(path, 'tic-tac-toe.data'), header = None)
