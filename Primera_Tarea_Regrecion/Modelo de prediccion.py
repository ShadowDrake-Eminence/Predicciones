
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""
Predicción de resultado entre Universidad de Chile y Deportes La Serena
Usando datos históricos, estadísticas de jugadores, goles, asistencias y tarjetas.
"""

# Cargar datos de partidos
df_uchile = pd.read_csv('Info_UChile/PartidosUCHILE.csv')
df_lserena = pd.read_csv('Info_LSerena/PartidosLSerena.csv')


# Cargar rendimiento de equipo (partidos)
# Los archivos de partidos ya contienen el rendimiento a nivel equipo

# Cargar datos analíticos individuales
analitica_uchile = pd.read_csv('Info_UChile/UChile_Analitica.csv')
analitica_lserena = pd.read_csv('Info_LSerena/Serena_Analitica.csv')



# Calcular promedios individuales (goles, puntos, asistencias, amarillas, rojas, edad, apariciones)
def resumen_individual(df):
    prom_goles = df['Goles'].replace('--', 0).astype(float).mean()
    prom_puntos = df['Puntos'].replace('--', 0).astype(float).mean()
    prom_asist = df['Asistencias'].replace('--', 0).astype(float).mean()
    prom_amarillas = df['Amarillas'].replace('--', 0).astype(float).mean()
    prom_rojas = df['Roja'].replace('--', 0).astype(float).mean()
    prom_edad = pd.to_numeric(df['Edad'], errors='coerce').fillna(0).mean()
    prom_apariciones = pd.to_numeric(df['Apariciones'], errors='coerce').fillna(0).mean()
    return prom_goles, prom_puntos, prom_asist, prom_amarillas, prom_rojas, prom_edad, prom_apariciones

uchile_goles_ind, uchile_puntos_ind, uchile_asist_ind, uchile_amarillas_ind, uchile_rojas_ind, uchile_edad_ind, uchile_apariciones_ind = resumen_individual(analitica_uchile)
lserena_goles_ind, lserena_puntos_ind, lserena_asist_ind, lserena_amarillas_ind, lserena_rojas_ind, lserena_edad_ind, lserena_apariciones_ind = resumen_individual(analitica_lserena)

# Cargar datos analíticos individuales
analitica_uchile = pd.read_csv('Info_UChile/UChile_Analitica.csv')
analitica_lserena = pd.read_csv('Info_LSerena/Serena_Analitica.csv')

# Calcular promedios individuales
def resumen_individual(df):
    prom_goles = df['Goles'].replace('--', 0).astype(float).mean()
    prom_puntos = df['Puntos'].replace('--', 0).astype(float).mean()
    return prom_goles, prom_puntos

uchile_goles_ind, uchile_puntos_ind = resumen_individual(analitica_uchile)
lserena_goles_ind, lserena_puntos_ind = resumen_individual(analitica_lserena)

# Unir datos de partidos directos entre ambos equipos
df_directos = pd.concat([
    df_uchile[(df_uchile['Visitante'] == 'La Serena') | (df_uchile['Local'] == 'La Serena')],
    df_lserena[(df_lserena['Visitante'] == 'Universidad de Chile') | (df_lserena['Local'] == 'Universidad de Chile')]
])

# Variables predictoras: localía, goles previos, asistencias, tarjetas
df_directos['Local_UChile'] = (df_directos['Local'] == 'Universidad de Chile').astype(int)
df_directos['Goles_UChile'] = np.where(df_directos['Local'] == 'Universidad de Chile', df_directos['Goles_Local'], df_directos['Goles_Visitante'])
df_directos['Goles_LSerena'] = np.where(df_directos['Local'] == 'La Serena', df_directos['Goles_Local'], df_directos['Goles_Visitante'])


# Agregar promedios individuales
df_directos['UChile_Goles_Ind'] = uchile_goles_ind


# Preparar dataset con todos los partidos de ambos equipos


def preparar_partidos(df, equipo, goles_ind, puntos_ind, asist_ind, amarillas_ind, rojas_ind, edad_ind, apariciones_ind):
    df = df.copy()
    df['Es_Local'] = (df['Local'] == equipo).astype(int)
    df['Goles_Equipo'] = np.where(df['Local'] == equipo, df['Goles_Local'], df['Goles_Visitante'])
    df['Goles_Rival'] = np.where(df['Local'] == equipo, df['Goles_Visitante'], df['Goles_Local'])
    df['Goles_Ind'] = goles_ind
    df['Puntos_Ind'] = puntos_ind
    df['Asist_Ind'] = asist_ind
    df['Amarillas_Ind'] = amarillas_ind
    df['Rojas_Ind'] = rojas_ind
    df['Edad_Ind'] = edad_ind
    df['Apariciones_Ind'] = apariciones_ind
    # Competencia internacional (1 si es Libertadores/Sudamericana, 0 si no)
    df['Comp_Internacional'] = df['Competencia'].str.contains('Libertadores|Sudamericana', case=False, na=False).astype(int)
    return df[['Es_Local', 'Goles_Equipo', 'Goles_Rival', 'Goles_Ind', 'Puntos_Ind', 'Asist_Ind', 'Amarillas_Ind', 'Rojas_Ind', 'Edad_Ind', 'Apariciones_Ind', 'Comp_Internacional']]

df_uchile_modelo = preparar_partidos(df_uchile, 'Universidad de Chile', uchile_goles_ind, uchile_puntos_ind, uchile_asist_ind, uchile_amarillas_ind, uchile_rojas_ind, uchile_edad_ind, uchile_apariciones_ind)
df_lserena_modelo = preparar_partidos(df_lserena, 'La Serena', lserena_goles_ind, lserena_puntos_ind, lserena_asist_ind, lserena_amarillas_ind, lserena_rojas_ind, lserena_edad_ind, lserena_apariciones_ind)

# Unir ambos datasets
df_modelo = pd.concat([df_uchile_modelo, df_lserena_modelo], ignore_index=True)



# Modelo de predicción usando todos los partidos y variables individuales
X = df_modelo[['Es_Local', 'Goles_Rival', 'Goles_Ind', 'Puntos_Ind', 'Asist_Ind', 'Amarillas_Ind', 'Rojas_Ind', 'Edad_Ind', 'Apariciones_Ind', 'Comp_Internacional']]
y = df_modelo['Goles_Equipo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print('--- Resultados del modelo de predicción ---')
print(f'Raíz del error cuadrático medio (RMSE): {rmse:.2f}')
print(f'Coeficiente de determinación (R²): {r2:.2f}')



# Ejemplo de predicción para un partido nuevo
print('\n--- Predicción para un partido nuevo entre UChile y La Serena ---')
# Supongamos que UChile juega de local y La Serena tiene rendimiento promedio
nuevo_partido = pd.DataFrame({
    'Es_Local': [1],
    'Goles_Rival': [np.mean(df_lserena_modelo['Goles_Equipo'])],
    'Goles_Ind': [uchile_goles_ind],
    'Puntos_Ind': [uchile_puntos_ind],
    'Asist_Ind': [uchile_asist_ind],
    'Amarillas_Ind': [uchile_amarillas_ind],
    'Rojas_Ind': [uchile_rojas_ind],
    'Edad_Ind': [uchile_edad_ind],
    'Apariciones_Ind': [uchile_apariciones_ind],
    'Comp_Internacional': [0] # 0 si no es internacional, 1 si lo es
})
pred_nuevo = model.predict(nuevo_partido)
print(f'Predicción de goles UChile en el próximo partido: {pred_nuevo[0]:.2f}')

print(f"Resultados de la evaluación:")
print(f"RMSE: {rmse:.2f} (error promedio de {rmse:.0f} goles)")
print(f"R-Cuadrado (R^2): {r2:.2f} (El modelo explica el {r2:.0%} de la variación de los goles)")
# Quedo mas o menos mal pero cumple.

