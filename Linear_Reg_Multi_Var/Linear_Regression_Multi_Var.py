print("Linear Regression with Multiple Variables: y = m1x1 + m2x2 +m3x3 + b")
print("Dependent Variabe: y")
print("Independent Variables (or) Features: x1, x2, x3")
print("C0-efficient: m1, m2, m3")
print("Y-Intercept: b")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import math

df = pd.read_csv("D:\\Python\\Programs\\Csv_Files\\homeprices_Multi.csv")
print(df)

Mean = df.bedrooms.mean()
Median = df.bedrooms.median()
Mode = df.bedrooms.mode()

Median = math.floor(Median)

print("Mean: ", Mean)
print("Median: ", Median)
print("Mode: ", Mode)

df.bedrooms.fillna(Median, inplace=True)

print(df.bedrooms)

# plt.xlabel("Area in sq.fts")
# plt.ylabel("Price in Rs")
# plt.scatter(df.area, df.price, marker='*')
# plt.show()

reg = linear_model.LinearRegression() #Object Creation
temp = reg.fit(df[['area', 'bedrooms', 'age']], df.price) #Independent Variables, Dependent Variables
print(temp)
m = reg.coef_
b = reg.intercept_
print(m)
print(b)

predict_price = reg.predict([[3000, 3, 40]]) #Sq.fts: 3000, Bedrooms: 3, Age: 40
print(predict_price)

predict_price = reg.predict([[2500, 4, 5]]) #Sq.fts: 3000, Bedrooms: 3, Age: 40
print(predict_price)

y = ((m[0] * 3000) + (m[1] * 3) + (m[2] * 40)) + b
y1 = ((m[0] * 2500) + (m[1] * 4) + (m[2] * 5)) + b
print(y)
print(y1)
