#1 Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2 Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#3 Training the Linear Regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#4 Training the Polynomial Regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#5 Visualising the Linear Regression results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Truth or Bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show

#6 Visualising the Polynomial Regression Results
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X_poly ),color='blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show
# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#7 Predicting a new result with Linear Regression
pred_lin = lin_reg.predict([[6.5]])
print(pred_lin)

#8 Predicting a new result with Polynomial Regression
pred_poly = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(pred_poly)
