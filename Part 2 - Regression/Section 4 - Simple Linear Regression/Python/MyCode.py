#1 Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#2 Importing Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

#3 Split the data set into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#4 Training the Simple Linear Regression Model on the Training Set
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)

#5 Predicting the Test Set Results
preds = lr.predict(X_test)

#6 Visualising the Training Set Results
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,lr.predict(X_train),color='green')

plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

#6 Visualising the Testing Set Results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_test,preds,color='green')

plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()

# plt.plot(X_train,lr.predict(X_train),color='green')=plt.plot(X_test,preds,color='green')
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,lr.predict(X_train),color='green')

plt.title('Salary vs Experience (Testing Set)')
plt.xlabel('Years of Experiance')
plt.ylabel('Salary')
plt.show()



