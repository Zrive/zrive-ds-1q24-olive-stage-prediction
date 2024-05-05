import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error
import logging

# Configurar el logging
logging.basicConfig(filename='nan_values.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def metrics(y: pd.Series, y_pred: pd.Series):
    """
    Calculates both accuracy and mean squared error (MSE) for a model based on series of actual and predicted values.
    The predicted values are rounded before calculating accuracy.

    Parameters:
    y (pd.Series): Series containing the actual values.
    y_pred (pd.Series): Series containing the predicted values.

    Returns:
    dict: Dictionary with the model's accuracy and MSE values.
    """

    # Check for NaN values and log them
    nan_indices_y = y[y.isna()].index
    nan_indices_y_pred = y_pred[y_pred.isna()].index
    if not nan_indices_y.empty or not nan_indices_y_pred.empty:
        logging.info(f'NaN found in y at indices: {nan_indices_y.tolist()}')
        logging.info(f'NaN found in y_pred at indices: {nan_indices_y_pred.tolist()}')

    # Proceed with calculations on non-NaN data only
    mask = y.notna() & y_pred.notna()
    y_filtered = y[mask]
    y_pred_filtered = y_pred[mask]

    y_pred_rounded = y_pred_filtered.round()
    accuracy_result = accuracy_score(y_filtered, y_pred_rounded)
    mse_result = mean_squared_error(y_filtered, y_pred_filtered)

    return {'accuracy': accuracy_result, 'mse': mse_result}

# Example usage:
# metrics(test_set['y'], test_set['y_pred'])
