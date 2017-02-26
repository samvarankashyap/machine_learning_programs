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


# lets see the data 
# we have two categorical values Country and Purchased in current dataset
# we have to encode all the categorical values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# tip X[rows: colums]
# though we have encoded its difficult for us to have a binary encoding is helpful
X[:, 0] = labelencoder_X.fit_transform(X[:,0])
print(X)
# 0 --> the column to work on 
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
# this makes the categorical variables as follows :
# three columns for three country and 0, or 1 signifying the country
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print(Y)

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

# feature scaling 
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
