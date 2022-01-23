print("Linear Regression with Single Variables: y=mx+b")
print("Dependent Variable or Price: y")
print("Independent Variables or Features: x")
print("Slope or Gradient or Co-efficient: m")
print("Y- Intercept: b")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("D:\\Python\\Programs\\Csv_Files\\Area_Price_2.csv") #Maintain Your path and keep the file
print(type(df))
print(df)

# plt.scatter(df.area, df.price, color='red', marker='+')
# plt.xlabel("Area in Sq.ft")
# plt.ylabel("Cost in Rs")
# plt.show()

new_df = df.drop('price', axis='columns') #Getting Areas data in the new dataframe
price = df.price
print(new_df)
print(price)

#Starting of Linear Regression with y=mx+b

reg = linear_model.LinearRegression() #Object creation
temp = reg.fit(new_df, price) #Fitting or Training
print(temp)

print("Prediction for 3300 Sq.fts: \n")
#Prediction of 3300 Sq.fts
temp = reg.predict([[3300]]) #Prediction Value
print(temp)
#y = mx+b
m = reg.coef_ #Co-efficient value (m: Slope or Gradient)
print(m)
b = reg.intercept_ #Intercept value (b: Y- Intercept)
print(b)

y = ((m * 3300)+ b)
print(y)

print("Prediction for 4800 Sq.fts: \n")
#Prediction of 4800 Sq.fts
temp = reg.predict([[4800]])
m = reg.coef_
b = reg.intercept_
y = ((m * 4800) + b)
print(temp)
print(m)
print(b)
print(y)

plt.xlabel("Area in Sq.ft")
plt.ylabel("Prediction Values")
plt.scatter(df.area, df.price, color='red')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.show()

d = pd.read_csv("D:\\Python\\Programs\\Csv_Files\\areas.csv") #Fetching Areas for prediction
print(d)
temp = reg.predict(d)
print(temp)
d['prices'] = temp
print(d['prices'])
d.to_csv('D:\\Python\\Programs\\Csv_Files\\Prediction.csv', index=False) #storing Prediction values

import pickle
with open('model_pickle', 'wb') as f:
    pickle.dump(reg, f)

with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)

print(mp.predict([[7200]]))

import joblib
joblib.dump(reg, 'model_joblib')
mj = joblib.load('model_joblib')

print(mj.predict([[8000]]))
