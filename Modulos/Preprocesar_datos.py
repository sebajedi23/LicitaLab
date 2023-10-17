import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import statsmodels.tsa.filters.hp_filter as hp
from datetime import datetime, timedelta
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from numpy.linalg import LinAlgError
import warnings
from dateutil.relativedelta import relativedelta
from itertools import combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import math
import joblib
import inspect
import os


class Preprocesar_datos():
  #Cargar las variables al modelo
  def __init__(self ,df_rubro, var_exogena, lag,inicio_timeseries = pd.to_datetime('2013-01-01')):
    self.df_rubro = df_rubro
    self.var_exogena = var_exogena
    self.lag = lag
    self.inicio_timeseries = inicio_timeseries

  #Procesamiento variables exogenas

  #PIB
  def procesar_pib(self):
    df_PIB = self.var_exogena[0].copy()
    #Crear más filas para todos los meses
    PIB_mensual = df_PIB.reindex(np.repeat(df_PIB.index.values, 3)).reset_index(drop=True)
    #Eliminar columna periodo
    PIB_mensual = PIB_mensual.drop('Periodo', axis = 1)
    #Crear una columna para cada mes llamada periodo
    PIB_mensual['Periodo'] = pd.date_range(start='2011-01-01', periods=len(PIB_mensual), freq='M')
    #Columna de interes: Mes y año
    PIB_mensual['Mes_Año'] = pd.to_datetime(PIB_mensual['Periodo']).dt.to_period('M')
    #Eliminar columna perido
    PIB_mensual = PIB_mensual.drop('Periodo', axis = 1)
    return PIB_mensual

  #IPC
  def procesar_ipc(self):
    df_IPC = self.var_exogena[1].copy()
    #Eliminar el último índice que corresponde a abril 2023
    df_IPC = df_IPC.drop(index = 147)
    df_IPC = df_IPC.drop(index = 148)
    ##Columna de interes: Mes y año
    df_IPC['Mes_Año'] = pd.to_datetime(df_IPC['Periodo']).dt.to_period('M')
    #Eliminar columna perido
    df_IPC = df_IPC.drop('Periodo', axis = 1)
    return df_IPC

  #IMACEC
  def procesar_imacec(self):
    df_IMACEC = self.var_exogena[2].copy()
    #Eliminar el último índice que corresponde a abril 2023
    df_IMACEC = df_IMACEC.drop(index = 147)
    ##Columna de interes: Mes y año
    df_IMACEC['Mes_Año'] = pd.to_datetime(df_IMACEC['Periodo']).dt.to_period('M')
    #Elimino columna perido
    df_IMACEC = df_IMACEC.drop('Periodo', axis = 1)

    return df_IMACEC

  # UTM
  def procesar_utm(self):

      df_utm = self.var_exogena[3].copy()
      df_utm['Mes_Año'] = pd.to_datetime(df_utm['Periodo']).dt.to_period('M')
      return df_utm[['Mes_Año', '1.Unidad tributaria mensual (UTM)']]

  # UF
  def procesar_uf(self):

      df_uf = self.var_exogena[4].copy()
      df_uf['Mes_Año'] = pd.to_datetime(df_uf['Periodo']).dt.to_period('M')
      return df_uf[['Mes_Año', '1.Unidad de fomento (UF)']]

  # Dólar
  def procesar_dolar(self):

      df_dolar = self.var_exogena[5].copy()
      df_dolar['Mes_Año'] = pd.to_datetime(df_dolar['Periodo']).dt.to_period('M')
      return df_dolar[['Mes_Año', '1.Dólar observado']]

  # EUR
  def procesar_eur(self):

      df_eur = self.var_exogena[6].copy()
      df_eur.drop(columns=['Vol.'],inplace=True)
      df_eur['Máximo'] = df_eur['Máximo'].astype(str).apply(lambda x: x.replace('.', '', 1).replace('.', ',', 1))
      df_eur['Máximo'] = df_eur['Máximo'].str.replace(',', '.').astype(float)
      df_eur['Mínimo'] = df_eur['Mínimo'].astype(str).str.replace(',', '.').astype(float)


      df_eur['Euro'] = (df_eur['Máximo'] + df_eur['Mínimo']) / 2

      df_eur['Mes_Año'] = pd.to_datetime(df_eur['Fecha']).dt.to_period('M')
      return df_eur[['Mes_Año', 'Euro']]

  # Fusionar todas las variables exogenas
  def merge_exogenas(self):
      df_pib = self.procesar_pib()
      df_ipc = self.procesar_ipc()
      df_imacec = self.procesar_imacec()
      df_merge = pd.merge(df_pib, df_ipc, on='Mes_Año', how='inner')
      df_merge.columns = ['PIB', 'Mes_Año', 'IPC']

      df_merge = pd.merge(df_merge, df_imacec, on='Mes_Año', how='inner')
      df_merge.columns = ['PIB', 'Mes_Año', 'IPC', 'IMACEC']

      return df_merge

  #Variables divisa
  def df_divisas(self):
      df_utm = self.procesar_utm()
      df_uf = self.procesar_uf()
      df_dolar = self.procesar_dolar()
      df_eur = self.procesar_eur()
      df_merge = pd.merge(df_utm, df_uf, on='Mes_Año', how='left')
      df_merge = pd.merge(df_merge, df_dolar, on='Mes_Año', how='left')
      df_merge = pd.merge(df_merge, df_eur, on='Mes_Año', how='left')

      return df_merge

  def seleccion_var_endogena(self):

    data = self.df_rubro.copy()
    data1 = data[['RubroN3', 'FechaCreacion', 'cantidad', 'totalLineaNeto', 'monedaItem']]
    # Convertir la columna 'FechaCreacion' en data1 a tipo de datos datetime
    data1['FechaCreacion'] = pd.to_datetime(data1['FechaCreacion'])
    data1['Mes_Año'] = pd.to_datetime(data1['FechaCreacion']).dt.to_period('M')
    # Filtrar las filas que tengan una fecha igual o posterior al 1 de enero de 2011
    data1 = data1[data1['FechaCreacion'] >= pd.to_datetime('2012-06-01')]

    return data1


  def procesamiento_data(self):
    data = self.seleccion_var_endogena()

    # Fusionar con la columna 'Periodo' de UTM para obtener la tasa de conversión correspondiente a cada fecha
    data = data.merge(self.df_divisas(), left_on='Mes_Año', right_on='Mes_Año', how='left')

    data['totalLineaNeto'] = pd.to_numeric(data['totalLineaNeto'], errors='coerce')
    data['1.Unidad tributaria mensual (UTM)'] = pd.to_numeric(data['1.Unidad tributaria mensual (UTM)'], errors='coerce')
    data['1.Unidad de fomento (UF)'] = pd.to_numeric(data['1.Unidad de fomento (UF)'], errors='coerce')
    data['1.Dólar observado'] = pd.to_numeric(data['1.Dólar observado'], errors='coerce')
    data['Euro'] = pd.to_numeric(data['Euro'], errors='coerce')


    data['total_convertido'] = np.where(data['monedaItem'] == 'UTM', data['totalLineaNeto'] * data['1.Unidad tributaria mensual (UTM)'],
                                        np.where(data['monedaItem'] == 'CLF', data['totalLineaNeto'] * data['1.Unidad de fomento (UF)'],
                                                  np.where(data['monedaItem'] == 'EUSD', data['totalLineaNeto'] * data['1.Dólar observado'],
                                                          np.where(data['monedaItem'] == 'EUR', data['totalLineaNeto'] * data['Euro'],
                                                                    np.where(data['monedaItem'] == 'CLP', data['totalLineaNeto'], np.nan)))))

    df = data[['RubroN3', 'FechaCreacion', 'cantidad','total_convertido']]
    df['cantidad'] = df['cantidad'].str.replace(',', '.').astype(float)
    # Agrupar por rubro, mes y año, y sumar las columnas 'cantidad' y 'total_convertido'
    df = df.groupby(['RubroN3', pd.Grouper(key='FechaCreacion', freq='M')]).agg({'cantidad': 'sum', 'total_convertido': 'sum'}).reset_index()

    # Obtener el rango completo de fechas
    fecha_minima = df['FechaCreacion'].min()
    fecha_maxima = df['FechaCreacion'].max()
    rango_fechas = pd.date_range(start=fecha_minima, end=fecha_maxima, freq='M')

    # Obtener todas las combinaciones posibles de "RubroN3" y el rango de fechas
    combinaciones = list(itertools.product(df['RubroN3'].unique(), rango_fechas))

    # Crear un DataFrame con todas las combinaciones
    df_combinaciones = pd.DataFrame(combinaciones, columns=['RubroN3', 'FechaCreacion'])

    # Combinar el DataFrame con todas las combinaciones con el DataFrame original
    df_rellenado = pd.merge(df_combinaciones, df, on=['RubroN3', 'FechaCreacion'], how='left')

    # Rellenar con ceros los valores faltantes en las columnas 'cantidad' y 'total_convertido'
    df_rellenado['cantidad'].fillna(0, inplace=True)
    df_rellenado['total_convertido'].fillna(0, inplace=True)

    # Ordenar el DataFrame resultante por 'RubroN3' y 'FechaCreacion'
    df_rellenado.sort_values(['RubroN3', 'FechaCreacion'], inplace=True)

    # Restablecer el índice del DataFrame resultante
    df_rellenado.reset_index(drop=True, inplace=True)
    df = df_rellenado.copy()
    return df

  def aplicar_lag(self,to_pred=0):

    df = self.procesamiento_data()

    # Convertir la columna "FechaCreacion" a tipo datetime
    df['FechaCreacion'] = pd.to_datetime(df['FechaCreacion'])

    # Extraer el mes y año de la columna "FechaCreacion"
    df['mes_año'] = df['FechaCreacion'].dt.to_period('M')

    if to_pred == 1:
        # Create a new DataFrame with 3 months forward for 'FechaCreacion' and 'mes_año'
        df_pred = df[['RubroN3', 'FechaCreacion', 'mes_año']].copy()
        df_pred['FechaCreacion'] = df_pred['FechaCreacion'] + pd.DateOffset(months=self.lag)

        # Set the 'FechaCreacion' to the last day of the month
        df_pred['FechaCreacion'] = df_pred['FechaCreacion'] + pd.offsets.MonthEnd(0)

        df_pred['mes_año'] = df_pred['FechaCreacion'].dt.to_period('M')
        #df_pred['totalLineaNeto'] = np.nan

        # Merge the new DataFrame with the original DataFrame based on 'RubroN3', 'FechaCreacion', and 'mes_año'
        df = pd.merge(df, df_pred, on=['RubroN3', 'FechaCreacion'], how='outer')

        # Drop duplicates created during the merge
        #df = df.drop_duplicates(subset=['RubroN3', 'FechaCreacion'], keep='first')


        # Calcular el desfase de X meses atrás
        df= df.rename(columns={'mes_año_x' : 'mes_año'})
        df.drop(columns='mes_año_y',inplace=True)
    # Extraer el mes y año de la columna "FechaCreacion"
    df['mes_año'] = df['FechaCreacion'].dt.to_period('M')

    df['mes_año_desfase'] = df['mes_año'].apply(lambda x: x - self.lag) #lag definidio al inicio
    # Combinar las bases de datos utilizando la columna 'mes_año_desfase' en lugar de 'mes_año'
    df_final = pd.merge(df, self.merge_exogenas(), left_on='mes_año_desfase', right_on='Mes_Año', how='left')
    # Eliminar las columnas de desfase y mes_año_desfase que ya no son necesarias
    df_final = df_final.drop(['mes_año_desfase', 'Mes_Año'], axis=1)

    # Ordenar el DataFrame por 'RubroN3' y 'FechaCreacion' en orden descendente
    df_final = df_final.sort_values(['RubroN3', 'FechaCreacion'])
    # Aplicar el mismo desfase de lag meses para la variable 'totalLineaNeto'
    df_final['total_convertido_desfase'] = df_final['total_convertido'].shift(self.lag)

    # Eliminar la columna original 'total_convertido'
    df_final = df_final.drop('total_convertido', axis=1)

    # Renombrar la columna 'total_convertido_desfase' como 'total_convertido'
    df_final = df_final.rename(columns={'total_convertido_desfase': 'total_convertido'})

    # Convertir la columna 'FechaCreacion' a tipo de datos datetime
    df_final['FechaCreacion'] = pd.to_datetime(df_final['FechaCreacion'])

    #Filtrar las filas que tengan una fecha igual o posterior a la fecha indicada como inicio de la serie de tiempo.
    df_final_final = df_final[df_final['FechaCreacion'] >= self.inicio_timeseries]
    df_final_final.columns.values[7] = 'totalLineaNeto'
    #######################
    ## LAG DE LA DEMANDA ##
    #######################

    # Ordenar el DataFrame por clase y fecha
    data_final = df_final_final.sort_values(['RubroN3', 'FechaCreacion'])

    # Agregar una nueva columna 'Lag Cantidad' con el lag de tres cuatro y cinco meses
    data_final[f'Lag {self.lag} Cantidad'] = data_final.groupby('RubroN3')['cantidad'].shift(self.lag)
    data_final[f'Lag {self.lag+1} Cantidad'] = data_final.groupby('RubroN3')['cantidad'].shift(self.lag+1)
    data_final[f'Lag {self.lag+2} Cantidad'] = data_final.groupby('RubroN3')['cantidad'].shift(self.lag+2)


    # Definir variables de año y mes
    data_final['Año'] = data_final['FechaCreacion'].dt.year
    data_final['Mes'] = data_final['FechaCreacion'].dt.month

    #Añadimos data del los promedios de montos transados en mercado publico
    monto_transado = self.var_exogena[7].copy()
    data_final = data_final.merge(monto_transado[['Mes', 'Promedio gasto acumulado', 'Promedio Monto transado']], left_on='Mes', right_on='Mes', how='left')
    data_final = data_final.drop(columns=['mes_año','IMACEC','Promedio gasto acumulado'])

    # Eliminar la columna 'cantidad' si to_pred es igual a 1
    if to_pred == 1:
        data_final = data_final.drop('cantidad', axis=1)
        pass
    data_final = data_final.fillna(0)
    return data_final
