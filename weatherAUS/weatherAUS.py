import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# Leer data
url = 'weatherAUS.csv'
data = pd.read_csv(url)


# Tratamiento de la data
data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainTomorrow.replace(['Yes', 'No'], [1, 0], inplace=True)
data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)


# Dividir la data en dos
data_train = data[:38767]
data_test = data[38767:]

x = np.array(data_train.drop(['RainTomorrow'], axis=1))
y = np.array(data_test.RainTomorrow)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['RainTomorrow'], axis=1))
y_test_out = np.array(data_test.RainTomorrow)