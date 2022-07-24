# Script de Preparación de Datos
###################################

import pandas as pd
import numpy as np
import os


# Leemos los archivos xlsx
def read_file_xlsx(filename):
    df = pd.read_excel(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Leemos los archivos csv
def read_file_csv(filename):
    df = pd.read_csv(os.path.join('../data/raw/', filename))
    print(filename, ' cargado correctamente')
    return df


# Exportamos la matriz de datos con las columnas seleccionadas
def data_exporting(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join('../data/processed/', filename))
    print(filename, 'exportado correctamente en la carpeta processed')


# Generamos las matrices de datos que se necesitan para la implementación
def main():
    # Matriz de Entrenamiento
    df1 = read_file_csv('poliza_train.csv')
    data_exporting(df1, ['edad','AniosDireccion','Gastocoche','Aniosempleo','Aniosresiden','ingres'],'poliza_train.csv')
    # Matriz de Validación
    df2 = read_file_csv('poliza_val.csv')
    data_exporting(df2, ['edad','AniosDireccion','Gastocoche','Aniosempleo','Aniosresiden','ingres'],'poliza_val.csv')
    # Matriz de Scoring
    df3 = read_file_csv('poliza_scor.csv')
    data_exporting(df3, ['edad','AniosDireccion','Gastocoche','Aniosempleo','Aniosresiden'],'poliza_scor.csv')


if __name__ == "__main__":
    main()
