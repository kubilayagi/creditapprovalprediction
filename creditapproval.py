# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
from scipy.stats import zscore

## IMPORT DATA
dataset = pd.read_csv('approvaldata.csv')
## PROCESS DATA
dataset.Approved.replace(('+', '-'), (1, 0), inplace=True)
dataset.Male.replace(('a', 'b'), (1, 0), inplace=True)
dataset.PriorDefault.replace(('t', 'f'), (1, 0), inplace=True)
dataset.Employed.replace(('t', 'f'), (1, 0), inplace=True)
dataset.DriversLicense.replace(('t', 'f'), (1, 0), inplace=True)
dataset.drop(['EducationLevel', 'Ethnicity', 'Citizen', 'ZipCode', 'Married'], axis=1, inplace=True)
dataset = dataset.loc[(dataset['BankCustomer'] == 'g') | (dataset['BankCustomer'] == 'p')]
dataset[['Income', 'CreditScore', 'Debt']] = dataset[['Income', 'CreditScore', 'Debt']].apply(zscore)
# TODO: bank customer needs to be determined, i feel like most people generally dont apply for cards at their own bank
