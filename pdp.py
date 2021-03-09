from IPython.display import display 
from pandas_profiling import ProfileReport

import pandas as pd


train = pd.read_csv('atmacup10_dataset/train.csv')
hp = pd.read_csv('atmacup10_dataset/historical_person.csv')
profile = ProfileReport(hp)
profile.to_file("profile_hp.html")