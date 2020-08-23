import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

## IMPORT DATA
dataset = pd.read_csv('approvaldata.csv')
## PROCESS DATA
dataset.Approved.replace(('+', '-'), (1, 0), inplace=True)
dataset.Male.replace(('a', 'b'), (1, 0), inplace=True)
dataset.PriorDefault.replace(('t', 'f'), (1, 0), inplace=True)
dataset.Employed.replace(('t', 'f'), (1, 0), inplace=True)
dataset.DriversLicense.replace(('t', 'f'), (1, 0), inplace=True)
dataset.drop(['EducationLevel', 'Ethnicity', 'Citizen', 'ZipCode', 'Married', 'DriversLicense'], axis=1, inplace=True)
dataset = dataset.loc[(dataset['BankCustomer'] == 'g') | (dataset['BankCustomer'] == 'p')]
dataset[['Income', 'CreditScore', 'Debt']] = dataset[['Income', 'CreditScore', 'Debt']].apply(zscore)
# TODO: bank customer needs to be determined, i feel like most people generally dont apply for cards at their own bank
print(dataset[['BankCustomer', 'Approved']].groupby(['BankCustomer', 'Approved']).size())
print('--------------')
print(dataset.loc[(dataset['Approved'] == 0) & (dataset['CreditScore'] > 0) & (dataset['Income'] > 0)])
dataset.replace('?', np.NaN, inplace=True)
dataset.dropna(inplace=True)
# temporarily
dataset.drop(['BankCustomer'], axis=1, inplace=True)

## SPLIT DATA
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# I don't think we need to scale here because I already did zcore
# sc = MinMaxScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

## BUILD AND TRAIN LOGISTIC REGRESSION MODEL
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

## TEST ON TEST DATA
y_pred = clf.predict(X_test)
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

## COMPUTE CONFUSION MATRIX
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

## COMPUTE ACCURACY SCORE
print(accuracy_score(y_test, y_pred))
# 0.8484848484848485

## VISUALIZE RESULTS