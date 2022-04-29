import numpy as np
import pandas as pd

# Leer data
url = 'weatherAUS.csv'
data = pd.read_csv(url)


# Tratamiento de la data
data.RainToday.replace(['Yes', 'No'], [1, 0], inplace=True)
data.RainTomorrow.replace(['Yes', 'No'], [1, 0], inplace=True)
data.drop(['Date', 'Location', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm'], axis=1, inplace=True)
data.dropna(axis=0, how='any', inplace=True)