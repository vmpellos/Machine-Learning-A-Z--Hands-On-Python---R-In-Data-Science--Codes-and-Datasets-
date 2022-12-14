#1 Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 Importing Dataset

dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values


#3 Taking Care of Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,
                        strategy='mean')
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#4 Encoding Categorical Data
#independent data column 0 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],
                       remainder='passthrough')
X= np.array(ct.fit_transform(X))

#Dependent data - class column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#5 Split the data set into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#6 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])
