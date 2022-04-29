import numpy as np
import pandas as pd

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