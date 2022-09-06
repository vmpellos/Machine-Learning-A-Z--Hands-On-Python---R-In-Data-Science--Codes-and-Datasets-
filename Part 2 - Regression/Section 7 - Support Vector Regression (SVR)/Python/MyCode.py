#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print(X)
print(y) # I have to reshape y. Must be an array (vertically printed)
y = y.reshape(len(y),1)
print(y)

#3 Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X  = sc_X.fit_transform(X)
y  = sc_y.fit_transform(y)

print(X)
print("\n")
print(y)


#4 Training the SVR model on the whole dataset
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')
svr.fit(X,y)


#5 Predicting a new Result
pred = svr.predict(sc_X.transform([[6.5]]))
print(pred)
pred = sc_y.inverse_transform(pred)
print(pred)


#6 Vsualising the SVR Results
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')

plt.plot(sc_X.inverse_transform(X),sc_y.inverse_transform(svr.predict(X)),color='blue')

plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#6 Visualising with better resolution and smoother curve
X_grid = np.arange(min(sc_X.inverse_transform(X)),max(sc_X.inverse_transform(X)),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(sc_X.inverse_transform(X),sc_y.inverse_transform(y),color='red')
plt.plot(X_grid,sc_y.inverse_transform(svr.predict(sc_X.transform(X_grid))),color='blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()


