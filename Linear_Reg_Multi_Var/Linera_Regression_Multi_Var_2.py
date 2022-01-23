import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

print("Independent Variables: experience, test_score(out of 10), interview_score(out of 10)")
print("Dependent Variable: Salary")

df = pd.read_csv("D:\\Python\\Programs\\Csv_Files\\hiring.csv")
print(df)

df['experience'].fillna('Two', inplace=True)
print(df)

i = 0
for num in df['experience']:
    temp = w2n.word_to_num(num)
    print(temp)
    df.at[i, 'experience'] = temp
    i += 1
    # print("i: ", i)

print(df)

Median = df['test_score(out of 10)'].median()
print(Median)

df.at[6, 'test_score(out of 10)'] = Median
print(df)

reg = linear_model.LinearRegression()
temp = reg.fit(df[["experience", "test_score(out of 10)", "interview_score(out of 10)"]], df["salary($)"])

print(temp)

# y = m1x1 + m2x2 + m3x3 + b

m = reg.coef_
b= reg.intercept_

Predict_Value = reg.predict([[2, 9, 6]])
Predict_Value_1 = reg.predict([[12, 10, 10]])
print(Predict_Value)
print(Predict_Value_1)

y = ((m[0] * 2) + (m[1] * 9) + (m[2] * 6) + b)
y1 = ((m[0] * 12) + (m[1] * 10) + (m[2] * 10) + b)
print(y)
print(y1)

plt.figure(figsize = (16, 12))

plt.subplot(2, 2, 1)
plt.xlabel("salary($)")
plt.ylabel("experience")
plt.title("salary($) vs experience")
plt.plot(df["salary($)"], df["experience"], marker="*")

plt.subplot(2, 2, 2)
plt.xlabel("salary($)")
plt.ylabel("test_score(out of 10)")
plt.title("salary($) vs test_score(out of 10)")
plt.plot(df["salary($)"], df["test_score(out of 10)"], marker="+")

plt.subplot(2, 2, 3)
plt.xlabel("salary($)")
plt.ylabel("interview_score(out of 10)")
plt.title("salary($) vs interview_score(out of 10)")
plt.plot(df["salary($)"], df["interview_score(out of 10)"], marker="^")

plt.savefig("D:\\Python\\Programs\\Images\\Linear_Reg_3.png")
plt.show()
