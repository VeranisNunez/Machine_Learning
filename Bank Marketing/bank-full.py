import numpy as np
import pandas as pd

# Leer data
url = 'bank-full.csv'
data = pd.read_csv(url)

# Tratamiento de la data
data.housing.replace(['yes', 'no'], [0, 1], inplace=True)
data.marital.replace(['married', 'single', 'divorced'],[0, 1, 2], inplace=True)
data.y.replace(['yes', 'no'], [0, 1], inplace=True)
data.education.replace(['primary', 'secondary', 'tertiary', 'unknown'],[0, 1, 2, 3], inplace=True)
data.default.replace(['yes', 'no'], [0, 1], inplace=True)
data.loan.replace(['yes', 'no'], [0, 1], inplace=True)
data.contact.replace(['unknown', 'cellular', 'telephone'],[0, 1, 2], inplace=True)
data.poutcome.replace(['unknown', 'success', 'failure', 'other'],[0, 1, 2, 3], inplace=True)
data.drop(['balance', 'duration', 'campaign', 'pdays', 'previous', 'job', 'day', 'month'], axis=1, inplace=True)

age_mean = data.age.mean()
data.age.replace(np.nan, age_mean, inplace=True)
ranges = [0, 8, 15, 18, 25, 40, 60, 100]
names = ['1', '2', '3', '4', '5', '6', '7']
data.age = pd.cut(data.age, ranges, labels=names)

data.dropna(axis=0, how='any', inplace=True)