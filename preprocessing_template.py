# Data PreProcessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Data.csv')
# all the columns except the last column from data.csv 
X = dataset.iloc[:, :-1].values
# choose the the last column of data.csv
Y = dataset.iloc[:, 3].values


# Splitting the dataset into training set and test set
from sklearn.cross_validation import train_test_split
# Good test size is 0.2, 0.3 
# 0.2 give 2 obs in test would be 2 and train has 8 records
# good random state is 42
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)
print(X_train)
print(X_test)
print(Y_train)
print(X_test)

"""
# feature scaling (optional)
# age is from 27 to 50
# salary is from 48k to 83k
# most of ml models are based on euclidain distances , so based on existing x and y the euclidain distance
# is dominated by the salary independent variable
# euclidian dist P1, P2 is sqrt((x2-x1)**2+(y2-y1)**2)
# two types of scaling 
# standardisation and normalization 
# xst = x - mean(x) / stddev(x)
# xnorm = x - min(x) / max(x)- min(x)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# for test set we need not fit we just transform
X_test = sc_X.transform(X_test)
print(X_train)
print(X_test)
# for y we need not apply feature scaling as its a classification problem 
# if y is in huge scale of values we need to do the feature scaling.
"""
