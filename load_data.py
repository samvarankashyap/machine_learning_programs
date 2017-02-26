import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Data.csv')
# all the columns except the last column from data.csv 
X = dataset.iloc[:, :-1].values
# choose the the last column of data.csv
Y = dataset.iloc[:, 3].values
