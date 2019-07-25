# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 21:35:26 2019

@author: rarro
"""

#Importing Libraries
#Mathematical tools
import numpy as np
#Chart plotting
import matplotlib.pyplot as plt
#Working with datasets
import pandas as pd

#Importing the dataset - save into same directory as data.csv
dataset = pd.read_csv('Data.csv')

#Creating matrix of independent variables
X = dataset.iloc[:, :-1].values #[lines of dataset:columns of dataset] (:-1 = all columns except last)

#Creating dependent variable 
Y = dataset.iloc[:,3].values #3 is the index of the dependent variable column

#Taking care of missing data
from sklearn.preprocessing import Imputer #Library for processing data  
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)

#fit imputer to our matrix x
imputer = imputer.fit(X[:,1:3]) #taking rows from column 1&2(missing values)
X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #Dummy vars class
# Begin with country variable
labelEncoder_X = LabelEncoder()
# fit object to the country data and save those values back into the correct variable in X
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])

#Creating dummy variables
oneHotEncoder = OneHotEncoder(categorical_features = [0]) # Categorical column
X = oneHotEncoder.fit_transform(X).toarray()

# We only need labelencoder for dependent variable (purchased)
labelEncoder_Y = LabelEncoder()
Y= labelEncoder_Y.fit_transform(Y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# (All of the data, fraction of data for test set, set random_state if you want to have same results as demo)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# Recompute sets to be scaled
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)