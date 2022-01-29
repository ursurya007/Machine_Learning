import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("titanic_data Data Set")
titanic_data = pd.read_csv("D:\\Python\\Programs\\Csv_Files\\titanic\\train.csv")
# print(titanic_data)
print(titanic_data.head())
print(titanic_data.info())
print(titanic_data.isnull())
#sns -- distplot, pairplot, joinplot, countplot, violinplot, boxplot, barplot
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

#count data between Survived and Deceased
sns.set_style('whitegrid')
sns.countplot(x="Survived", data=titanic_data)
plt.show()

#count data between Survived and Deceased comparsion with sex
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex', data=titanic_data, palette='RdBu_r')
plt.show()

#count data between Survived and Deceased comparsion with Passanger Class
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Pclass', data=titanic_data, palette='rainbow')
plt.show()

#Distplot of data with respect to the ages -- to fix Null data with our analysis
sns.distplot(titanic_data['Age'].dropna(), kde=False, color='darkred', bins=40)
plt.show()

#Histogram visualization of Ages
titanic_data['Age'].hist(bins=30, color='darkred', alpha=0.3)
plt.show()

#Count of Siblings and spouse
sns.countplot(x='SibSp',data=titanic_data)
plt.show()

#Fare details
titanic_data['Fare'].hist(color='green', bins=40, figsize=(8, 4))
plt.show()

# import cufflinks as cf
# cf.go_offline()

# titanic_data['Fare'].iplot(kind='hist', bins=30, color='green')

plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass', y='Age', data=titanic_data, palette='winter')
plt.show()

#function to fill the Null data of ages with respect to the Pclass
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age

titanic_data['Age'] = titanic_data[['Age','Pclass']].apply(impute_age,axis=1)

sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

titanic_data.drop('Cabin', axis=1, inplace=True)
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# values = {"Embarked": "Q"}
# titanic_data.fillna(value=values, axis=1, inplace=True)
titanic_data["Embarked"].fillna("Q", inplace=True)
sns.heatmap(titanic_data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

print(pd.get_dummies(titanic_data['Embarked'], drop_first=True).head())
sex = pd.get_dummies(titanic_data['Sex'], drop_first=True)
embark = pd.get_dummies(titanic_data['Embarked'], drop_first=True)

titanic_data.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
print(titanic_data.head())

#concatenating sex and embarked
titanic_data = pd.concat([titanic_data, sex, embark], axis=1)
print(titanic_data.head())

#Building Regression Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

#Y-axis -- Dependent Variable -- Survived
#X-axis -- Independent Variables -- PassengerId, Pclass,	Age,	SibSp,	Parch,	Fare
#Train data set = 70%, Test Set = 30%

X_train, X_test, y_train, y_test = train_test_split(titanic_data.drop('Survived', axis=1),
                                                    titanic_data['Survived'], test_size=0.30,
                                                    random_state=101)
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train) #Train the model

#getting Predicted output data
predictions = logmodel.predict(X_test)

accuracy = confusion_matrix(y_test, predictions) #Confussion Matrix calculation
print(accuracy)

accuracy = accuracy_score(y_test, predictions) #Comparing predicted value with test value
print(accuracy)
print(predictions)
print(classification_report(y_test, predictions))
