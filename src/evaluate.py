# Código de Evaluación - Modelo de Riesgo de Default en un Banco de Corea
############################################################################

import pandas as pd
import numpy as np
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import os


# Cargar la tabla transformada
def eval_model(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename)).set_index('ID')
    print(filename, ' cargado correctamente')
    # Leemos el modelo entrenado para usarlo
    package = '../models/best_model.pkl'
    model = pickle.load(open(package, 'rb'))
    print('Modelo importado correctamente')
    # Predecimos sobre el set de datos de validación 
    X_test = df[['edad', 'AniosDireccion', 'Gastocoche', 'Aniosempleo', 'Aniosresiden']]
    X_test = sm.add_constant(X_test)
    y_test = df[['ingres']]
    y_pred_test=model.predict(X_test)
    # Generamos métricas del modelo
    print('MAPE: ',      metrics.mean_absolute_error(y_test, y_pred_test))
    print('MAE: ',       metrics.mean_squared_error(y_test, y_pred_test))
    print('RMSE: ',  np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
    rmse_ols = np.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    print(f"El error (RMSE) de test es: {rmse_ols}")


# Validación desde el inicio
def main():
    df = eval_model('poliza_val.csv')
    print('Finalizó la validación del Modelo')


if __name__ == "__main__":
    main()