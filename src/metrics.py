import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def accuracy(test: pd.DataFrame, actual_col: str, pred_col: str):
    """
    Calcula la precisión de un modelo comparando las columnas de valores reales y predichos en un DataFrame.
    Los valores predichos son redondeados antes de calcular la precisión.

    Parámetros:
    test (pd.DataFrame): DataFrame que contiene las columnas de valores reales y predicciones.
    actual_col (str): Nombre de la columna en el DataFrame que contiene los valores reales.
    pred_col (str): Nombre de la columna en el DataFrame que contiene los valores predichos.

    Devuelve:
    float: El valor de precisión del modelo.
    """

    mask = test[actual_col].notna()
    test_filtered = test[mask]

    test_filtered['y_pred_rounded'] = test_filtered[pred_col].round()

    accuracy = accuracy_score(
        test_filtered[actual_col], test_filtered['y_pred_rounded'])

    return accuracy
# accuracy(test, 'y', 'y_pred')


def mse(test: pd.DataFrame, actual_col: str, pred_col: str):
    """
    Calcula el error cuadrático medio (MSE) para un modelo comparando las columnas de valores reales y predichos en un DataFrame.

    Parámetros:
    test (pd.DataFrame): DataFrame que contiene las columnas de valores reales y predicciones.
    actual_col (str): Nombre de la columna en el DataFrame que contiene los valores reales.
    pred_col (str): Nombre de la columna en el DataFrame que contiene los valores predichos.

    Devuelve:
    float: El valor de MSE del modelo.
    """

    mask = test[actual_col].notna()
    test_filtered = test[mask]

    mse = mean_squared_error(test_filtered[actual_col], test_filtered[pred_col])

    return mse
# mse(test, 'y', 'y_pred')
