#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#3 Encoding Categorical Data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],
                       remainder='passthrough')
X = np.array(ct.fit_transform(X))

#4 Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#5 Training the Multiple Liner Regression Model on the Training Set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

#6 Predicting the Test Results
preds = lr.predict(X_test)
np.set_printoptions(precision=2) # display numerical values with only 2 decimals after comma
print(np.concatenate((preds.reshape(len(preds),1),y_test.reshape(len(y_test),1)),axis=1))
