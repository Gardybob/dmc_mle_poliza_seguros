# Código de Entrenamiento - Modelo de Riesgo de Default en un Banco de Corea
############################################################################


import pandas as pd
import statsmodels.api as sm
import pickle
import os


# Cargar la tabla transformada
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/processed', filename))
    X_train = df[['edad', 'AniosDireccion', 'Gastocoche', 'Aniosempleo', 'Aniosresiden']]
    X_train = sm.add_constant(X_train)
    y_train = df[['ingres']]
    print(filename, ' cargado correctamente')
    # Entrenamos el modelo con toda la muestra
    M_R = sm.OLS(y_train,X_train)
    M_R = M_R.fit()
    print('Modelo entrenado')
    # Guardamos el modelo entrenado para usarlo en produccion
    package = '../models/best_model.pkl'
    pickle.dump(M_R, open(package, 'wb'))
    print('Modelo exportado correctamente en la carpeta models')


# Entrenamiento completo
def main():
    read_file_csv('poliza_train.csv')
    print('Finalizó el entrenamiento del Modelo')


if __name__ == "__main__":
    main()
