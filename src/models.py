import pandas as pd
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline


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


def baseline(train: pd.DataFrame, test: pd.DataFrame, target_column: str):
    """
    Baseline: Calculamos la media ponderada para el DataFrame de entrenamiento y aplicamos el mapeo resultante al conjunto de prueba.

    Parámetros:
    train (pd.DataFrame): DataFrame de entrenamiento con columnas 'year', 'estado_mayoritario y la columna objetivo especificada.
    test (pd.DataFrame): DataFrame de prueba con la columna 'estado_mayoritario'.
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
# baseline(train_set, test_set, 'y')


def logistic_regression_model(train: pd.DataFrame, test: pd.DataFrame, target_column: str, train_columns: list, penalty='l2', C=1.0, solver='liblinear'):
    """
    Train and evaluate a logistic regression model and return its coefficients.

    Parameters:
    - train (pd.DataFrame): The training dataset.
    - test (pd.DataFrame): The testing dataset.
    - target_column (str): The name of the target column.
    - train_columns (list): List of feature columns to train the model on.
    - penalty (str): Type of penalty to apply (l1 or l2).
    - C (float): Inverse of regularization strength; smaller values specify stronger regularization.
    - solver (str): Algorithm to use in the optimization problem.

    Returns:
    - pd.DataFrame: DataFrame containing coefficients and the test set with predictions.
    """

    # Create the pipeline for the logistic regression model
    model = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("logistic", LogisticRegression(penalty=penalty, C=C, solver=solver))
    ])

    # Train the model using the specified features and target column
    model.fit(train[train_columns], train[target_column])

    # Extract the logistic regression model from the pipeline
    logistic_model = model.named_steps['logistic']

    # Retrieve coefficients and feature names
    coefficients = logistic_model.coef_[0]
    features_and_coefs = pd.DataFrame({
        'Feature': train_columns,
        'Coefficient': coefficients
    })

    # Predict the probabilities of the positive class
    test['y_pred'] = model.predict(test[train_columns])

    return features_and_coefs, test
# features_and_coefs, test_predictions = logistic_regression_model(train_set, test_set, 'target', train_cols, penalty='l1', C=0.01, solver='liblinear')


def gradient_boosting_model(train: pd.DataFrame, test: pd.DataFrame, target_column: str, train_columns: list, n_estimators=100, learning_rate=0.1, max_depth=3):
    """
    Train and evaluate a Gradient Boosting model and return feature importances.

    Parameters:
    - train (pd.DataFrame): The training dataset.
    - test (pd.DataFrame): The testing dataset.
    - target_column (str): The name of the target column.
    - train_columns (list): List of feature columns to train the model on.
    - n_estimators (int): The number of boosting stages to be used in the model.
    - learning_rate (float): How much each tree contributes to the final result.
    - max_depth (int): The maximum depth of each individual tree.

    Returns:
    - pd.DataFrame: DataFrame containing feature importances and the test set with predictions.
    """

    # Crear el pipeline para el modelo Gradient Boosting
    model = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("gradient_boosting", GradientBoostingClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth))
    ])

    # Entrenar el modelo usando las características especificadas y la columna objetivo
    model.fit(train[train_columns], train[target_column])

    # Extraer el modelo de Gradient Boosting desde el pipeline
    boosting_model = model.named_steps['gradient_boosting']

    # Obtener importancias de características
    feature_importances = boosting_model.feature_importances_
    features_and_importances = pd.DataFrame({
        'Feature': train_columns,
        'Importance': feature_importances
    })

    # Ordenar por importancia
    features_and_importances = features_and_importances.sort_values(
        by='Importance', ascending=False)

    # Predecir las etiquetas para el conjunto de prueba
    test['y_pred'] = model.predict(test[train_columns])

    return features_and_importances, test


def gradient_boosting_model_gbm(train: pd.DataFrame, test: pd.DataFrame, target_column: str, train_columns: list, n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0) -> tuple:
    """
    Train and evaluate a Gradient Boosting Machine (GBM) model and return feature importances.

    Parameters:
    - train (pd.DataFrame): The training dataset.
    - test (pd.DataFrame): The testing dataset.
    - target_column (str): The name of the target column.
    - train_columns (list): List of feature columns to train the model on.
    - n_estimators (int): The number of boosting stages to be used in the model.
    - learning_rate (float): How much each tree contributes to the final result.
    - max_depth (int): The maximum depth of each individual tree.
    - subsample (float): The fraction of samples used for fitting each tree.

    Returns:
    - tuple: A tuple containing the DataFrame with feature importances and the test set with predictions.
    """

    # Create a pipeline with a standard scaler and the Gradient Boosting model
    model = Pipeline([
        ("standard_scaler", StandardScaler()),
        ("gradient_boosting", GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample
        ))
    ])

    # Train the model using the specified features and target column
    model.fit(train[train_columns], train[target_column])

    # Extract the Gradient Boosting model from the pipeline
    boosting_model = model.named_steps['gradient_boosting']

    # Get feature importances
    feature_importances = boosting_model.feature_importances_
    features_and_importances = pd.DataFrame({
        'Feature': train_columns,
        'Importance': feature_importances
    })

    # Sort by importance
    features_and_importances = features_and_importances.sort_values(
        by='Importance', ascending=False)

    # Predict labels for the test set
    test['y_pred'] = model.predict(test[train_columns])

    return features_and_importances, test
