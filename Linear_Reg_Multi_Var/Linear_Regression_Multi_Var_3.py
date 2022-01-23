import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, metrics

bostons = datasets.load_boston(return_X_y=False)

x = bostons.data
y = bostons.target

#matrix: x, response: y
print(x)
print(y)

#splitting x and y in to training and training sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)

m = reg.coef_
print("coef: ", m)
var_score = reg.score(x_test, y_test)
print("var score: {}".format(var_score))

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
            color="green", s=10, label='Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
            color="blue", s=10, label='Test data')

## plotting line for zero residual error
plt.hlines(y=0, xmin=0, xmax=50, linewidth=2)

## plotting legend
plt.legend(loc='upper right')

## plot title
plt.title("Residual errors")

## method call for showing the plot
plt.show()
