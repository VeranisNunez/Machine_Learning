import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

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


# Dividir la data en dos
data_train = data[:22604]
data_test = data[22604:]

x = np.array(data_train.drop(['y'], 1))
y = np.array(data_train.y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['y'], 1))
y_test_out = np.array(data_test.y)


## REGRESIÓN LOGÍSTICA

# Seleccionar un modelo
logreg = LogisticRegression(solver='lbfgs', max_iter = 7600)

# Entrenamiento del modelo
logreg.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Regresión Logística')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {logreg.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {logreg.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {logreg.score(x_test_out, y_test_out)}')


## MAQUINA DE SOPORTE VECTORIAL

# Seleccionar un modelo
svc = SVC(gamma='auto')

# Entrenamiento del modelo
svc.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Maquina de soporte vectorial')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {svc.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {svc.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {svc.score(x_test_out, y_test_out)}')


## ARBOL DE DECISIÓN

# Seleccionar un modelo
arbol = DecisionTreeClassifier()

# Entrenamiento del modelo
arbol.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Decisión Tree')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {arbol.score(x_train, y_train)}')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {arbol.score(x_test, y_test)}')

# Accuracy de Validación
print(f'accuracy de Validación: {arbol.score(x_test_out, y_test_out)}')


## RANDOM FOREST

# Seleccionar un modelo
random_forest = RandomForestClassifier()

# Entrenamiento del modelo
random_forest.fit(x_train, y_train)

# MÉTRICAS
print('*'*50)
print('Random Forest')

# Accuracy de Test de Entrenamiento
print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test, y_test)}')

# Accuracy de Entrenamiento de Entrenamiento
print(f'accuracy de Entrenamiento de Entrenamiento: {random_forest.score(x_train, y_train)}')

# Accuracy de validacion
print(f'accuracy de Test de Entrenamiento: {random_forest.score(x_test_out, y_test_out)}')