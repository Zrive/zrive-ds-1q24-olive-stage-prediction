from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
import pandas as pd


def metrics(y: pd.Series, y_pred: pd.Series):
    """
    Calculate the accuracy, mean squared error (MSE), and mean absolute error (MAE) for a model based on real
    and predicted value series. The predicted values are rounded before calculating accuracy.

    Parameters:
    y (pd.Series): Series containing the actual values.
    y_pred (pd.Series): Series containing the predicted values.

    Returns:
    dict: Dictionary containing model accuracy, MSE, and MAE values.
    """

    # Round predicted values to calculate accuracy
    y_pred_rounded = y_pred.round()
    accuracy_result = accuracy_score(y, y_pred_rounded)

    # Calculate Mean Squared Error (MSE)
    mse_result = mean_squared_error(y, y_pred)

    # Calculate Mean Absolute Error (MAE)
    mae_result = mean_absolute_error(y, y_pred)

    # Return the results in a dictionary
    return {
        'accuracy': accuracy_result,
        'mse': mse_result,
        'mae': mae_result
    }
