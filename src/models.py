import pandas as pd
import numpy as np


def train_test_split(df: pd.DataFrame, split_year: int = 2021, max_year: int = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Divide un DataFrame en conjuntos de entrenamiento y prueba basado en un año específico, con una opción para excluir
    años posteriores a un máximo dado.

    Parámetros:
    df (pd.DataFrame): El DataFrame a dividir, que debe contener una columna de fecha de tipo datetime.
    split_year (int): El año que se usará como límite para dividir los datos. El conjunto de entrenamiento
                      incluirá este año y años anteriores.
    max_year (int, opcional): El último año para incluir en el conjunto de prueba. Todos los datos posteriores a este año
                              serán excluidos del conjunto de prueba. Si no se especifica, se incluirán todos los datos posteriores a split_year.

    Devuelve:
    train (pd.DataFrame): DataFrame de entrenamiento que incluye el split_year y años anteriores.
    test (pd.DataFrame): DataFrame de prueba que incluye años después del split_year hasta el max_year, si se especifica.
    """

    df['year'] = df['fecha'].dt.year

    train = df[df['year'] <= split_year].copy()
    if max_year:
        test = df[(df['year'] > split_year) & (df['year'] <= max_year)].copy()
    else:
        test = df[df['year'] > split_year].copy()

    return train, test
# train_set, test_set = train_test_split(df, split_year=2021, max_year=2022)


def feature_label_split(df: pd.DataFrame, label_col: str):
    """
    Split DataFrame into features (X) and labels (y).
    """
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def baseline(train: pd.DataFrame, test: pd.DataFrame, target_column: str):
    """
    Baseline: Calculamos la media ponderada para el DataFrame de entrenamiento y aplicamos el mapeo resultante al conjunto de prueba.

    Parámetros:
    train (pd.DataFrame): DataFrame de entrenamiento con columnas 'year', 'estado_fenologico_unificado' y la columna objetivo especificada.
    test (pd.DataFrame): DataFrame de prueba con la columna 'estado_fenologico_unificado'.
    target_column (str): Nombre de la columna objetivo en el DataFrame que se usará para calcular la media ponderada.

    Devuelve:
    pd.DataFrame: DataFrame de prueba con las predicciones añadidas.
    """

    # Calcular el máximo año en el conjunto de entrenamiento para el descuento por antigüedad
    max_year = train['year'].max()
    train['weight'] = train['year'].apply(lambda x: 0.1 ** (max_year - x))

    # Agrupar por estado fenológico y año, y calcular la media ponderada
    grouped = train.groupby(['estado_mayoritario', 'year'])
    weighted_means = grouped.apply(lambda x: (
        x[target_column] * x['weight']).sum() / x['weight'].sum()).reset_index()
    weighted_means.rename(columns={0: 'weighted_mean'}, inplace=True)

    # Crear un diccionario con la media ponderada más reciente por estado fenológico
    mapeo_referencia = weighted_means.sort_values('year', ascending=False).drop_duplicates(
        'estado_mayoritario').set_index('estado_mayoritario')['weighted_mean'].to_dict()

    # Aplicar el mapeo de referencia al conjunto de prueba para obtener las predicciones
    test['y_pred'] = test['estado_mayoritario'].map(mapeo_referencia)

    return test
# baseline(train, test, 'y')
