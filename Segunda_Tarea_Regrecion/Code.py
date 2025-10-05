import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('Segunda_Tarea_Regrecion/Data.csv')

X = df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
y = df['Precio_UF']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('--- Resultados del modelo de regresión lineal ---')
print(f'Raíz del error cuadrático medio (RMSE): {rmse:.2f}')
print(f'Coeficiente de determinación (R²): {r2:.2f}')
print('\nPredicciones vs Valores reales:')
for real, pred in zip(y_test, y_pred):
    print(f'Real: {real:.2f} UF\tPredicción: {pred:.2f} UF')