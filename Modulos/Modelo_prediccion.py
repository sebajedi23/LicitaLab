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


class Modelo_prediccion():
  #Cargar las variables al modelo
  def __init__(self ,data_procesada, modelo=None, lag=3,inicio_timeseries = pd.to_datetime('2013-01-01'),modelos_entrenados = None):
    self.data_procesada = data_procesada.copy()
    self.modelo = modelo
    self.lag = lag
    self.inicio_timeseries = inicio_timeseries
    self.modelos_entrenados = modelos_entrenados
    #Variables temporales de los datos
    self.fecha_maxima = self.data_procesada.FechaCreacion.max()
    self.n_meses = lag
    self.fecha_resultante = self.fecha_maxima - timedelta(days=30 * self.n_meses)


  def verificar_inputs(self):
    if self.modelo is not None and self.modelos_entrenados is not None:
      print("Por favor, ingrese solo un modelo a entrenar o uno ya entrenado, pero no ambos.")
      return False

    if self.modelo is None and self.modelos_entrenados is None:
      print("Por favor, ingrese un modelo a entrenar o uno ya entrenado.")
      return False

    if self.modelo is not None and self.modelos_entrenados is None:
      print(f'Se ha seleccionado el modelo {str(self.modelo)} para ser entrenado')
      return True

    if self.modelo is None and self.modelos_entrenados is not None:
      if len(self.modelos_entrenados) == 3:
        print(f'Se ha seleccionado el modelo pre-entrenado {str(self.modelos_entrenados[2])}')
        return True
      else:
        print('Por favor ingrese una lista con el siguiente orden de modelos entrenados: [ARIMA, ARIMAX, Modelo_regresion]')
        return False

  def modelo_arima(self,ver_resultados=0):
    if not self.verificar_inputs():
      return

    # Obtener los rubros únicos
    rubros = self.data_procesada['RubroN3'].unique()
    # Crear un dataframe vacío para almacenar las predicciones totales
    predicciones_total = pd.DataFrame()
    modelos_arima = {}
    print()
    print('Comenzando modelo ARIMA')
    if self.modelo is not None: #Entrenamiento
      # Iteración sobre los rubros
      for rubro in rubros:
          # Filtrar los datos para el rubro actual y ordenarlos
          ex1 = self.data_procesada[self.data_procesada['RubroN3'] == rubro].sort_values('FechaCreacion')

          # Convertir la columna 'FechaCreacion' en el índice del dataframe
          ex1.set_index('FechaCreacion', inplace=True)

          inicio_test = self.fecha_resultante

          # Dividir los datos en entrenamiento y test. Se tomarán los últimos 3 meses para reducir el error como prueba
          train_data = ex1.loc[:inicio_test ]['cantidad']
          test_data = ex1.loc[inicio_test : ]['cantidad']

          # Búsqueda de órdenes (p, d, q) que minimicen el RMSE
          p_values = range(0, 12)
          d_values = range(0, 3)
          q_values = range(0, 3)

          rmse_results = []

          for p, d, q in itertools.product(p_values, d_values, q_values):
              order = (p, d, q)

              try:
                  # Ajustar el modelo ARIMA a los datos de entrenamiento con la orden actual
                  model = ARIMA(train_data, order=order)
                  model_fit = model.fit()

                  # Realizar una predicción en el rango completo de fechas
                  forecast = model_fit.predict(start=ex1.index[0], end=ex1.index[-1])
                  forecast = pd.DataFrame({
                  'Fecha': forecast.index,
                  'Prediccion': forecast.values
                  })

                  # Calcular el RMSE para las predicciones actuales
                  rmse = np.sqrt(mean_squared_error(ex1.loc[self.fecha_resultante : ]['cantidad'], forecast.query('Fecha >= @inicio_test')['Prediccion']))

                  # Agregar el RMSE a la lista de resultados
                  rmse_results.append((order, rmse))

              except LinAlgError:
                  continue

          # Verificar si se encontraron órdenes válidas
          if len(rmse_results) > 0:
              # Ordenar los resultados de RMSE de menor a mayor
              rmse_results.sort(key=lambda x: x[1])

              # Obtener la orden con el menor RMSE
              best_order = rmse_results[0][0]

              # Ajustar el modelo ARIMA con la mejor orden a todos los datos
              model = ARIMA(ex1['cantidad'], order=best_order)
              model_fit = model.fit()
              #Agregamos el modelo entrenado al diccionario
              modelos_arima[rubro] = model_fit
              # Realizar una predicción en el rango completo de fechas
              forecast = model_fit.predict(start=ex1.index[0], end=ex1.index[-1])

              # Crear un nuevo dataframe con las fechas y las predicciones
              predicciones = pd.DataFrame({
                  'Fecha': forecast.index,
                  'Prediccion': forecast.values
              })

              # Transformar los valores negativos a 0 en la columna 'Prediccion'
              predicciones['Prediccion'] = predicciones['Prediccion'].apply(lambda x: abs(x) if x < 0 else x)

              # Agregar una columna con el rubro correspondiente
              predicciones['Rubro'] = rubro

              # Concatenar las predicciones al dataframe total
              predicciones_total = pd.concat([predicciones_total, predicciones], ignore_index=True)
              # Calcular el RMSE solo para las predicciones
              rmse = np.sqrt(mean_squared_error(ex1.loc[inicio_test : ]['cantidad'], predicciones.query('Fecha >= @inicio_test')['Prediccion']))
              if ver_resultados == 0:
                print(f'Clase {rubro} completada!')
              if ver_resultados == 1:
                # Mostrar el gráfico de los datos reales y las predicciones
                plt.figure(figsize=(10, 6))
                plt.plot(ex1.loc['2013':].index, ex1.loc['2013':]['cantidad'], label='Datos reales')
                plt.plot(predicciones.query("Fecha >= '2013'")['Fecha'], predicciones.query("Fecha >= '2013'")['Prediccion'], label='Predicción ARIMA')
                plt.xlabel('Fecha')
                plt.ylabel('Cantidad')
                plt.title(rubro)
                plt.legend()
                plt.show()
                print('RMSE:', rmse)
      predicciones_total.columns = ['FechaCreacion', 'ARIMA', 'RubroN3']
      data2 = self.data_procesada.merge(predicciones_total, how='left', on=['FechaCreacion', 'RubroN3'])
      print('Arima completado!')
      return data2,modelos_arima

    #Predicción
    if self.modelos_entrenados is not None:
      model_arima = self.modelos_entrenados[0]
      # Iteración sobre los rubros
      for rubro in rubros:
          # Filtrar los datos para el rubro actual y ordenarlos
          ex1 = self.data_procesada[self.data_procesada['RubroN3'] == rubro].sort_values('FechaCreacion')

          # Convertir la columna 'FechaCreacion' en el índice del dataframe
          ex1.set_index('FechaCreacion', inplace=True)

          # Realizar una predicción en el rango completo de fechas con el modelo ya entrenado
          model_fit = model_arima[rubro]
          forecast = model_fit.predict(start=ex1.index[0], end=ex1.index[-1])

          # Crear un nuevo dataframe con las fechas y las predicciones
          predicciones = pd.DataFrame({
              'Fecha': forecast.index,
              'Prediccion': forecast.values
          })

          # Transformar los valores negativos a 0 en la columna 'Prediccion'
          predicciones['Prediccion'] = predicciones['Prediccion'].apply(lambda x: abs(0) if x < 0 else x)

          # Agregar una columna con el rubro correspondiente
          predicciones['Rubro'] = rubro

          # Concatenar las predicciones al dataframe total
          predicciones_total = pd.concat([predicciones_total, predicciones], ignore_index=True)
          if ver_resultados == 0:
            print(f'Clase {rubro} completada!')
          if ver_resultados == 1:
            # Mostrar el gráfico de los datos reales y las predicciones
            plt.figure(figsize=(10, 6))
            plt.plot(ex1.loc['2013':].index, ex1.loc['2013':]['cantidad'], label='Datos reales')
            plt.plot(predicciones.query("Fecha >= '2013'")['Fecha'], predicciones.query("Fecha >= '2013'")['Prediccion'], label='Predicción ARIMA')
            plt.xlabel('Fecha')
            plt.ylabel('Cantidad')
            plt.title(rubro)
            plt.legend()
            plt.show()
      predicciones_total.columns = ['FechaCreacion', 'ARIMA', 'RubroN3']
      data2 = self.data_procesada.merge(predicciones_total, how='left', on=['FechaCreacion', 'RubroN3'])
      print('Arima completado!')
      return data2


  def modelo_arimax(self, ver_resultados=0):
    # Crear un dataframe vacío para almacenar las predicciones totales
    predicciones_total_arimax = pd.DataFrame()
    rubros = self.data_procesada['RubroN3'].unique()
    if self.modelo is not None: #Entrenamiento
      data2,modelos_arima = self.modelo_arima()
      print('')
      print('Comenzando modelo ARIMAX')
      # Lista para almacenar los resultados de RMSE
      rmse_results = []
      #Diccionario para almacenar modelos ARIMAX
      modelos_arimax = {}

      # Iteración sobre los rubros
      for rubro in rubros:
          # Filtrar los datos para el rubro actual y ordenarlos
          ex1 = data2[data2['RubroN3'] == rubro].sort_values('FechaCreacion')

          # Convertir la columna 'FechaCreacion' en el índice del dataframe
          ex1.set_index('FechaCreacion', inplace=True)
          inicio_test = self.fecha_resultante - relativedelta(months=3)
          # Dividir los datos en entrenamiento (hasta 2022) y prueba (2022-2023)
          train_data = ex1.loc[:inicio_test]['cantidad']
          test_data = ex1.loc[inicio_test: ]['cantidad']
          exog_train = ex1.loc[:inicio_test][['PIB', 'IPC', 'totalLineaNeto', 'Año', 'Mes', f'Lag {self.lag} Cantidad', f'Lag {self.lag+1} Cantidad', f'Lag {self.lag+2} Cantidad', 'Promedio Monto transado', 'ARIMA']]
          exog_test = ex1.loc[inicio_test:][['PIB', 'IPC', 'totalLineaNeto', 'Año', 'Mes', f'Lag {self.lag} Cantidad', f'Lag {self.lag+1} Cantidad', f'Lag {self.lag+2} Cantidad', 'Promedio Monto transado', 'ARIMA']]
          #exog_test2 = ex1.loc['2023'][['PIB', 'IPC', 'totalLineaNeto', 'Año', 'Mes', f'Lag {lag} Cantidad', f'Lag {lag+1} Cantidad', f'Lag {lag+2} Cantidad', 'Promedio Monto transado', 'ARIMA']]
          # Búsqueda de órdenes (p, d, q) que minimicen el RMSE
          p_values = range(0, 12)
          d_values = range(0, 3)
          q_values = range(0, 3)

          rmse_results = []

          for p, d, q in itertools.product(p_values, d_values, q_values):
              order = (p, d, q)

              try:
                # Ajustar el modelo ARIMAX con el orden actual
                model = ARIMA(train_data, order=order, exog=exog_train)
                model_fit = model.fit()
                # Realizar la predicción en el rango de fechas de prueba
                forecast = model_fit.predict(start=ex1.index[0], end=ex1.index[-1], exog=exog_test)
                    # Crear un nuevo dataframe con las fechas y las predicciones
                forecast = pd.DataFrame({
                'Fecha': forecast.index,
                'Prediccion': forecast.values
                })
                # Calcular el RMSE para las predicciones actuales
                rmse = np.sqrt(mean_squared_error(ex1.loc[inicio_test: ]['cantidad'], forecast.query('Fecha >= @inicio_test')['Prediccion']))

                # Agregar el RMSE a la lista de resultados
                rmse_results.append((order, rmse))
              except LinAlgError:
                continue
          if len(rmse_results) > 0:

            # Ordenar los resultados de RMSE de menor a mayor
            rmse_results.sort(key=lambda x: x[1])

            # Obtener los órdenes que minimizan el RMSE
            best_order = rmse_results[0][0]
            best_rmse = rmse_results[0][1]

            # Ajustar el modelo ARIMAX con los mejores órdenes
            train_data = ex1.loc[:]['cantidad']
            exog_train = ex1.loc[:][['PIB', 'IPC', 'totalLineaNeto', 'Año', 'Mes', f'Lag {self.lag} Cantidad', f'Lag {self.lag+1} Cantidad', f'Lag {self.lag+2} Cantidad', 'Promedio Monto transado', 'ARIMA']]
            exog_test = ex1.loc[inicio_test:][['PIB', 'IPC', 'totalLineaNeto', 'Año', 'Mes', f'Lag {self.lag} Cantidad', f'Lag {self.lag+1} Cantidad', f'Lag {self.lag+2} Cantidad', 'Promedio Monto transado', 'ARIMA']]

            model = ARIMA(train_data, order=best_order, exog=exog_train)
            model_fit = model.fit()
            modelos_arimax[rubro] = model_fit
            # Realizar una predicción en el rango completo de fechas
            forecast = model_fit.predict(start=ex1.index[0], end=ex1.index[-1], exog=exog_test)

            # Crear un nuevo dataframe con las fechas y las predicciones
            predicciones = pd.DataFrame({
                'FechaCreacion': forecast.index,
                'ARIMAX': forecast.values
            })

            # Transformar los valores negativos a 0 en la columna 'ARIMAX'
            predicciones['ARIMAX'] = predicciones['ARIMAX'].apply(lambda x: abs(x) if x < 0 else x)

            # Agregar una columna con el rubro correspondiente
            predicciones['RubroN3'] = rubro

            # Concatenar las predicciones al dataframe total
            predicciones_total_arimax = pd.concat([predicciones_total_arimax, predicciones], ignore_index=True)

            # Calcular el RMSE solo para las predicciones
            rmse = np.sqrt(mean_squared_error(ex1.loc[inicio_test:]['cantidad'], predicciones.query('FechaCreacion >= @inicio_test')['ARIMAX']))

            if ver_resultados == 0:
              print(f'Clase {rubro} completada!')
            if ver_resultados == 1:

              # Mostrar el gráfico de los datos reales y las predicciones
              plt.figure(figsize=(10, 6))
              plt.plot(ex1.loc['2013':].index, ex1.loc['2013':]['cantidad'], label='Datos reales')
              plt.plot(predicciones.query("FechaCreacion >= '2013'")['FechaCreacion'], predicciones.query("FechaCreacion >= '2013'")['ARIMAX'], label='Predicción ARIMAX')
              plt.xlabel('Fecha')
              plt.ylabel('Cantidad')
              plt.title(rubro)
              plt.legend()
              plt.show()

              print('RMSE:', rmse)

            # Agregar el RMSE a la lista de resultados
            rmse_results.append(rmse)

      predicciones_total_arimax.columns = ['FechaCreacion','ARIMAX','RubroN3']
      data3= data2.merge(predicciones_total_arimax,how='left',on=['FechaCreacion','RubroN3'])
      print('Modelo ARIMAX finalizado')
      print()
      return data3, modelos_arima, modelos_arimax


    if self.modelo is None: #Prediccion
      data2 = self.modelo_arima()
      print()
      print('Comenzando modelo ARIMAX')
      model_arimax = self.modelos_entrenados[1]
      # Iteración sobre los rubros
      for rubro in rubros:
          # Filtrar los datos para el rubro actual y ordenarlos
          ex1 = data2[data2['RubroN3'] == rubro].sort_values('FechaCreacion')

          # Convertir la columna 'FechaCreacion' en el índice del dataframe
          ex1.set_index('FechaCreacion', inplace=True)
          exog_test = ex1.loc[:][['PIB', 'IPC', 'totalLineaNeto', 'Año', 'Mes', f'Lag {self.lag} Cantidad', f'Lag {self.lag+1} Cantidad', f'Lag {self.lag+2} Cantidad', 'Promedio Monto transado', 'ARIMA']]

          model_fit = model_arimax[rubro]
          # Realizar una predicción en el rango completo de fechas
          forecast = model_fit.predict(start = ex1.index[0], end= ex1.index[-1], exog=exog_test)

          # Crear un nuevo dataframe con las fechas y las predicciones
          predicciones = pd.DataFrame({
              'FechaCreacion': forecast.index,
              'ARIMAX': forecast.values
          })

          # Transformar los valores negativos a 0 en la columna 'ARIMAX'
          predicciones['ARIMAX'] = predicciones['ARIMAX'].apply(lambda x: abs(x) if x < 0 else x)

          # Agregar una columna con el rubro correspondiente
          predicciones['RubroN3'] = rubro

          # Concatenar las predicciones al dataframe total
          predicciones_total_arimax = pd.concat([predicciones_total_arimax, predicciones], ignore_index=True)

          if ver_resultados == 0:
            print(f'Clase {rubro} completada!')
          if ver_resultados == 1:

            # Mostrar el gráfico de los datos reales y las predicciones
            plt.figure(figsize=(10, 6))
            plt.plot(ex1.loc[:].index, ex1.loc[:]['cantidad'], label='Datos reales')
            plt.plot(predicciones['FechaCreacion'], predicciones['ARIMAX'], label='Predicción ARIMAX')
            plt.xlabel('Fecha')
            plt.ylabel('Cantidad')
            plt.title(rubro)
            plt.legend()
            plt.show()

      predicciones_total_arimax.columns = ['FechaCreacion','ARIMAX','RubroN3']
      data3= data2.merge(predicciones_total_arimax,how='left',on=['FechaCreacion','RubroN3'])
      print('Modelo ARIMAX finalizado')
      print()
      return data3

  def guardar_modelos(self, modelo,rubro):
      if modelo is not None:
          nombre = f'modelo_{rubro}.pkl'
          with open(nombre, 'wb') as f:
              joblib.dump(modelo, f)
          #print('Modelos entrenados guardados exitosamente.')
          return nombre

  def cargar_modelos(self,rubro):
      try:
          nombre = f'modelo_{rubro}.pkl'
          with open(nombre, 'rb') as f:
              modelos = joblib.load(f)
          print('Modelos entrenados cargados exitosamente.')
          return modelos
      except FileNotFoundError:
          print('No se encontró ningún modelo entrenado previamente.')
          return None

  def aplicar_modelo_prediccion(self,cols= ['Mes','Año'],ver_resultados=0,feat_select = None,results_arima_arimax=0):
    print('Comenzando modelo de regresión')
    all_predictions = []
    modelos_predicciones = {}
    features_selected = {}
    predicciones_total_demanda = pd.DataFrame()
    scalers = {}
    if self.modelo is not None: #Entrenamiento
      data,arima_pred,arimax_pred = self.modelo_arimax(ver_resultados=results_arima_arimax)

      # Calcular la desviación estándar agrupando por "RubroN3"
      desviacion_estandar = data.groupby('RubroN3')['cantidad'].std().reset_index()
      desviacion_estandar.columns = ['RubroN3', 'sd_nyear']

      # Fusionar el resultado de la desviación estándar con el DataFrame original
      data = data.merge(desviacion_estandar, on='RubroN3', how='left')

      data.drop(columns=cols, inplace=True)

      train_data = data[data['FechaCreacion'] <= self.fecha_resultante + timedelta(days=31)]
      test_data = data[data['FechaCreacion'] >= self.fecha_resultante]
      graf_data = data[data['FechaCreacion'] <= (self.fecha_maxima - timedelta(days=30 * (self.lag+1)))]

      rubros = data['RubroN3'].unique()

      total_rmse_train = 0.0
      total_rmse_test = 0.0
      total_mae_train = 0.0
      total_mae_test = 0.0
      total_r2_train = 0.0
      total_r2_test = 0.0

      resultados_df = pd.DataFrame(columns=['RubroN3', 'RMSE', 'RMSE / demanda_promedio_testeo'])


      for rubro in rubros:
          train_rubro_data = train_data[train_data['RubroN3'] == rubro]
          test_rubro_data = test_data[test_data['RubroN3'] == rubro]
          graf_rubro_data = graf_data[graf_data['RubroN3'] == rubro]

          X_train = train_rubro_data.drop(['cantidad', 'RubroN3', 'FechaCreacion','sd_nyear'], axis=1)
          y_train = train_rubro_data['cantidad']

          X_test = test_rubro_data.drop(['cantidad', 'RubroN3', 'FechaCreacion','sd_nyear'], axis=1)

          scaler = StandardScaler()
          X_train_scaled = scaler.fit_transform(X_train)
          X_test_scaled = scaler.transform(X_test)

          scalers[rubro] = scaler
          model = self.modelo

          model.fit(X_train_scaled, y_train)

          attribute_names = X_train.columns

          best_r2_score = -float('inf')
          best_rmse_score = float('inf')
          best_selected_features = None

          for r in range(2, len(attribute_names) + 1):
              for combination in combinations(attribute_names, r):
                  selected_features = list(combination)

                  X_train_selected = X_train[selected_features]
                  X_test_selected = X_test[selected_features]

                  model.fit(X_train_selected, y_train)

                  y_pred_train = model.predict(X_train_selected)
                  y_pred_test = model.predict(X_test_selected)

                  rmse_test = mean_squared_error(test_rubro_data['cantidad'], y_pred_test, squared=False)
                  r2_test = r2_score(test_rubro_data['cantidad'], y_pred_test)

                  if r2_test >= best_r2_score and rmse_test <= best_rmse_score:
                      best_r2_score = r2_test
                      best_rmse_score = rmse_test
                      best_selected_features = selected_features

          X_train_best = X_train[best_selected_features]
          X_test_best = X_test[best_selected_features]

          model= self.modelo
          model_fit = model.fit(X_train_best, y_train)

          # Guardar los modelos seleccionados en un archivo usando joblib
          modelos_predicciones[rubro] = self.guardar_modelos(model_fit,rubro)


          y_pred_train_best = model_fit.predict(X_train_best)
          y_pred_test_best = model_fit.predict(X_test_best)

          for i in range(len(y_pred_test_best)):
              if y_pred_test_best[i] < 0:
                  y_pred_test_best[i] = 0
          train_rubro_data['prediccion'] = y_pred_train_best
          test_rubro_data['prediccion'] = y_pred_test_best
          all_predictions.append(test_rubro_data)

          data = pd.concat([train_rubro_data, test_rubro_data], axis=0)


          rmse_train_best = mean_squared_error(train_rubro_data['cantidad'], y_pred_train_best, squared=False)
          rmse_test_best = mean_squared_error(test_rubro_data['cantidad'], y_pred_test_best, squared=False)
          mae_train_best = mean_absolute_error(train_rubro_data['cantidad'], y_pred_train_best)
          mae_test_best = mean_absolute_error(test_rubro_data['cantidad'], y_pred_test_best)
          r2_train_best = r2_score(train_rubro_data['cantidad'], y_pred_train_best)
          r2_test_best = r2_score(test_rubro_data['cantidad'], y_pred_test_best)

          demanda_promedio_testeo = test_rubro_data['cantidad'].mean()
          rmse_div_demanda_promedio_testeo = rmse_test_best / demanda_promedio_testeo

          total_rmse_train += rmse_train_best
          total_rmse_test += rmse_test_best
          total_mae_train += mae_train_best
          total_mae_test += mae_test_best
          total_r2_train += r2_train_best
          total_r2_test += r2_test_best

          resultados_df = pd.concat([resultados_df, pd.DataFrame({'RubroN3': rubro, 'RMSE': rmse_test_best, 'RMSE / demanda_promedio_testeo': rmse_div_demanda_promedio_testeo}, index=[0])], ignore_index=True)

          if ver_resultados == 1:
            plt.figure(figsize=(12, 6))
            #plt.plot(graf_rubro_data.query('FechaCreacion >= 2020')['FechaCreacion'], graf_rubro_data.query('FechaCreacion >= 2020')['cantidad'], label='Data')
            plt.plot(train_rubro_data.query('FechaCreacion >= 2020')['FechaCreacion'], train_rubro_data.query('FechaCreacion >= 2020')['cantidad'], label='Datos de Entrenamiento')
            plt.plot(test_rubro_data['FechaCreacion'], test_rubro_data['cantidad'], label='Datos de Prueba')
            plt.plot(test_rubro_data['FechaCreacion'], y_pred_test_best, label='Predicciones')
            plt.xlabel('Fecha de Creación')
            plt.ylabel('Cantidad')
            plt.title(f'Rubro: {rubro}')
            plt.legend()
            plt.show()

            print("")
            print(f'Los features seleccionados para el rubro {rubro} son:')
            print(best_selected_features)
          features_selected[rubro] = best_selected_features

          print()
          print('RMSE Test', rubro + ':' ,rmse_test_best)
          print('MAE Test', rubro + ':', mae_test_best)
          print('R2 Test', rubro + ':', r2_test_best)

      predicciones_df = pd.concat(all_predictions, ignore_index=True)

      avg_rmse_train = total_rmse_train / len(rubros)
      avg_rmse_test = total_rmse_test / len(rubros)
      avg_mae_train = total_mae_train / len(rubros)
      avg_mae_test = total_mae_test / len(rubros)
      avg_r2_train = total_r2_train / len(rubros)
      avg_r2_test = total_r2_test / len(rubros)

      print()
      print('------------------------')
      print('Métricas Promedio:')
      print(f'RMSE (Prueba): {avg_rmse_test}')
      print(f'MAE (Prueba): {avg_mae_test}')
      print(f'R2 Score (Prueba): {avg_r2_test}')
      print('-------------------------')

      print()
      print('Tabla de Resultados:')
      print(resultados_df)
      print('Modelo de regresión finalizado')
      return predicciones_df, features_selected , arima_pred, arimax_pred, scalers, resultados_df

    if self.modelo is None:
      data = self.modelo_arimax(ver_resultados=results_arima_arimax)
      data.drop(columns=cols, inplace=True)
      if 'sd_nyear' in data.columns:
        data.drop(columns= 'sd_nyear')
      rubros = data['RubroN3'].unique()
      scalers = self.modelos_entrenados[2]
      for rubro in rubros:
          data2 = data.query('RubroN3 == @rubro')

          X = data2[feat_select[rubro]]
          #X = X.drop(columns='FechaCreacion')
          #print(X.columns)
          data2.set_index('FechaCreacion',inplace=True)
          scaler = scalers[rubro]
          #X_scaled = scaler.transform(X)

          model_fit = self.cargar_modelos(rubro)

          prediccion = model_fit.predict(X)
          # Crear un nuevo dataframe con las fechas y las predicciones
          predicciones = pd.DataFrame({
              'FechaCreacion': data2.index,
              'Prediccion_demanda': prediccion
          })
          # Transformar los valores negativos a 0 en la columna 'ARIMAX'
          predicciones['Prediccion_demanda'] = predicciones['Prediccion_demanda'].apply(lambda x: abs(x) if x < 0 else x)
          # Agregar una columna con el rubro correspondiente
          predicciones['RubroN3'] = rubro
          #print(predicciones)
          # Concatenar las predicciones al dataframe total
          predicciones_total_demanda = pd.concat([predicciones_total_demanda, predicciones], ignore_index=True)

          if ver_resultados == 1:
            plt.figure(figsize=(12, 6))
            plt.plot(predicciones['FechaCreacion'], predicciones['Prediccion_demanda'], label='Prediccion demanda')
            plt.xlabel('Fecha de Creación')
            plt.ylabel('Cantidad')
            plt.title(f'Rubro: {rubro}')
            plt.legend()
            plt.show()
          if ver_resultados == 0:
            print(f'Predicción de demanda de la clase {rubro} completa!')

      #predicciones_df = pd.concat(all_predictions, ignore_index=True)
      predicciones_total_demanda.columns = ['FechaCreacion','Prediccion_demanda','RubroN3']
      data3= data.merge(predicciones_total_demanda,how='left',on=['FechaCreacion','RubroN3'])
      data_f = pd.DataFrame()
      for rubro in rubros:
        data_final = data3.query('RubroN3 == @rubro')
        data_final = data_final.iloc[-3:]
        data_f= pd.concat([data_f,data_final])
      print('Modelo de regresión finalizado') 
      return data_f
