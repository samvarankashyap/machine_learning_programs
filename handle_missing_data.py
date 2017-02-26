import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/Data.csv')
# all the columns except the last column from data.csv 
X = dataset.iloc[:, :-1].values
# choose the the last column of data.csv
Y = dataset.iloc[:, 3].values

# Handling Missing data.
# option 1 we can remove the records with missing data 
# option 2 replace the missing data with mean of the value

# sklearn -> Scilearn kit 
from sklearn.preprocessing import Imputer
# missing_values = 'NaN'
# axis = 0 --> columns , 
# strategy = mean --> replace with mean , we can also take median or mode etc 
imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
#fit the imputer  on all rows on columns 1 to 2
# ie., age and salary column
imputer = imputer.fit(X[:, 1:3])
# now replace the missing data
X[:, 1:3] = imputer.transform(X[:, 1:3])



