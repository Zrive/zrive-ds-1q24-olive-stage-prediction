import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a file handler and set the logging level
file_handler = logging.FileHandler('evaluation_metrics1.log')
file_handler.setLevel(logging.INFO)

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

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
    return {"accuracy": accuracy_result, "mse": mse_result}
# metrics(test_set['y'], test_set['y_pred'])


def evaluate_classification(y_true, y_pred) -> dict:
    metrics_dict = {}
    try:
        metrics_dict["accuracy"] = accuracy_score(y_true, y_pred)
        metrics_dict["mse"] = mean_squared_error(y_true, y_pred)
        metrics_dict["precision_for_class"] = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics_dict["recall_for_class"] = recall_score(
            y_true, y_pred, average=None, zero_division=0
        )
        metrics_dict["f1_for_class"] = f1_score(
            y_true, y_pred, average=None, zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)
        metrics_dict["confusion_matrix"] = cm
        return metrics_dict
    except Exception as e:
        logger.error("Error computing metrics: %s", str(e))

def log_metrics(model_name: str, metrics_dict: dict) -> None:
    """
    Log the computed metrics.
    """
    logger.info(f"Computed Metrics for {model_name}:")
    for metric, value in metrics_dict.items():
        if metric == "confusion_matrix":
            logger.info(f"{metric}:\n{value}")
        else:
            logger.info(f"{metric}: {value}")
    logger.info("----------------------------------")


def evaluate_configuration(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    evaluation_metrics: dict,
) -> None:
    # y_test_pred = model.predict_proba(X_test)[:, 1]
    # y_train_pred = model.predict_proba(X_train)[:, 1]

    y_test_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    evaluation_metrics[model_name] = {
        "train": evaluate_classification(y_train, y_train_pred),
        "test": evaluate_classification(y_test, y_test_pred),
    }
    log_metrics(model_name + "_train", evaluation_metrics[model_name]["train"])
    log_metrics(model_name + "_test", evaluation_metrics[model_name]["test"])


def evaluate_configuration_lightgbm(
    model: Pipeline,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    evaluation_metrics: dict,
) -> None:
    # y_test_pred = model.predict_proba(X_test)[:, 1]
    # y_train_pred = model.predict_proba(X_train)[:, 1]

    y_train_pred_prob = model.predict(X_train, num_iteration=model.best_iteration)
    y_train_pred = np.argmax(y_train_pred_prob, axis=1)

    y_test_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_test_pred = np.argmax(y_test_pred_prob, axis=1)

    evaluation_metrics[model_name] = {
        "train": evaluate_classification(y_train, y_train_pred),
        "test": evaluate_classification(y_test, y_test_pred),
    }
    log_metrics(model_name + "_train", evaluation_metrics[model_name]["train"])
    log_metrics(model_name + "_test", evaluation_metrics[model_name]["test"])