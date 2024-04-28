import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error


def metrics(y: pd.Series, y_pred: pd.Series):
    """
    Calcula tanto la precisión como el error cuadrático medio (MSE) para un modelo basado en series de valores reales
    y predichos. Los valores predichos son redondeados antes de calcular la precisión.

    Parámetros:
    y (pd.Series): Serie que contiene los valores reales.
    y_pred (pd.Series): Serie que contiene los valores predichos.

    Devuelve:
    dict: Diccionario con los valores de precisión y MSE del modelo.
    """

    # Asegurarse de que ambas series no tienen valores NaN donde se compara
    mask = y.notna() & y_pred.notna()
    y_filtered = y[mask]
    y_pred_filtered = y_pred[mask]

    # Redondear las predicciones para calcular la precisión
    y_pred_rounded = y_pred_filtered.round()
    accuracy_result = accuracy_score(y_filtered, y_pred_rounded)

    # Calcular el MSE con los valores predichos originales, no los redondeados
    mse_result = mean_squared_error(y_filtered, y_pred_filtered)

    # Devolver los resultados en un diccionario
    return {'accuracy': accuracy_result, 'mse': mse_result}
# metrics(test_set['y'], test_set['y_pred'])
