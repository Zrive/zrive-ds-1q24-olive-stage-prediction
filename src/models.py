import pandas as pd
import numpy as np


def train_test_split(df: pd.DataFrame, split_year: int = 2021, max_year: int = None) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits a DataFrame into training and testing sets based on a specific year, with an option to exclude years after a given maximum.

    Parameters:
    df (pd.DataFrame): The DataFrame to split, which must contain a datetime-type date column.
    split_year (int): The year that will be used as the cutoff for splitting the data. The training set
                      will include this year and previous years.
    max_year (int, optional): The last year to include in the test set. Any data following this year
                              will be excluded from the test set. If not specified, all data following split_year will be included.

    Returns:
    train (pd.DataFrame): Training DataFrame including split_year and previous years.
    test (pd.DataFrame): Testing DataFrame including years after split_year up to max_year, if specified.
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
    Baseline: We compute the weighted mean for the training DataFrame and apply the resulting mapping to the test set.

    This function uses a discounting approach to account for the age of the data, applying a decay factor of 0.8 to each year's data.
    This means each previous year's data is worth 80% of the next more recent year, reducing the impact of older data more gradually
    than more severe discounting would. This helps balance the influence of both recent and older entries in modeling.

    Parameters:
    train (pd.DataFrame): Training DataFrame with 'year', 'estado_fenologico_unificado', and the specified target column.
    test (pd.DataFrame): Testing DataFrame with the 'estado_fenologico_unificado' column.
    target_column (str): Name of the target column in the DataFrame to be used for calculating the weighted mean.

    Returns:
    pd.DataFrame: Testing DataFrame with the added predictions.
    """

    # Calculate the maximum year in the training set for age discounting
    max_year = train['year'].max()
    train['weight'] = train['year'].apply(lambda x: 0.8 ** (max_year - x))

    # Group by phenological state and year, and calculate the weighted mean
    grouped = train.groupby(['estado_fenologico_unificado', 'year'])
    weighted_means = grouped.apply(lambda x: (
        x[target_column] * x['weight']).sum() / x['weight'].sum()).reset_index()
    weighted_means.rename(columns={0: 'weighted_mean'}, inplace=True)

    # Create a dictionary with the most recent weighted mean by phenological state
    mapeo_referencia = weighted_means.sort_values('year', ascending=False).drop_duplicates(
        'estado_fenologico_unificado').set_index('estado_fenologico_unificado')['weighted_mean'].to_dict()

    # Apply the reference mapping to the test set to obtain predictions
    test['y_pred'] = test['estado_fenologico_unificado'].map(mapeo_referencia)

    return test

# baseline(train_set, test_set, 'y')
