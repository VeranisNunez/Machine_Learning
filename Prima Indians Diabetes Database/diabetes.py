import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Leer data
url = 'diabetes.csv'
data = pd.read_csv(url)


# Tratamiento de la data
data.drop(['Pregnancies', 'BloodPressure', 'SkinThickness'], axis= 1, inplace = True)
data.Age.replace(np.nan, 33, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60, 100]
nombres = ['1', '2', '3', '4', '5', '6', '7']
data.Age = pd.cut(data.Age, rangos, labels=nombres)
data.dropna(axis=0,how='any', inplace=True)


# Dividir la data en dos
data_train = data[:384]
data_test = data[384:]

x = np.array(data_train.drop(['Outcome'], axis=1))
y = np.array(data_test.Outcome)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], axis=1))
y_test_out = np.array(data_test.Outcome)