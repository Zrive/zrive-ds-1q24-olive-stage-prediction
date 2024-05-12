import pandas as pd
import numpy as np

TARGET_COLS = ['codparcela', 'fecha', 'estado_mayoritario']
SPINE_COLS = ['codparcela', 'fecha', 'target', 'campaña', 'estado_mayoritario']
METEO_COLUMNS = ['FAPAR', 'GNDVI', 'LST', 'NDVI', 'NDWI', 'SAVI', 'SIPI', 'SSM']


def generate_target(parcelas_df: pd.DataFrame, window_size: int = 14, window_tolerance: int = 2) -> pd.DataFrame:
    df = parcelas_df[TARGET_COLS].copy()
    df['fecha'] = df['fecha'].astype('datetime64[ns]')  # Set units in datetime format

    # Generate a date to search for PHENOLOGICAL_STATE in the future
    parcelas_df['fecha_futuro'] = parcelas_df['fecha'] + pd.Timedelta(days=window_size)

    # Merge the 2 DataFrames with a time delta tolerance (±2 days)
    parcelas_with_target_df = pd.merge_asof(
        parcelas_df.sort_values('fecha'), df.sort_values('fecha'),
        by='codparcela', left_on='fecha_futuro', right_on='fecha',
        suffixes=('', '_future'), direction='nearest', tolerance=pd.Timedelta(days=window_tolerance))

    # Generate target column - Number of phenological states that have passed in the selected window size
    parcelas_with_target_df['target'] = parcelas_with_target_df['estado_mayoritario_future'] - \
        parcelas_with_target_df['estado_mayoritario']

    # Drop all rows with NULL in the target
    parcelas_with_target_df = parcelas_with_target_df[parcelas_with_target_df['target'].notnull(
    )]

    # Set target to 0 when it's negative
    parcelas_with_target_df.loc[parcelas_with_target_df['target'] < 0, 'target'] = 0

    # Remove values from target greater than 3
    parcelas_with_target_df = parcelas_with_target_df[parcelas_with_target_df['target'] <= 3]

    return parcelas_with_target_df.sort_values(by=['codparcela', 'fecha'])


def create_spine(parcelas_df: pd.DataFrame) -> pd.DataFrame:

    target_df = generate_target(parcelas_df)

    # Check if the necessary columns are present
    missing_columns = [col for col in SPINE_COLS if col not in target_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Create the spine DataFrame by selecting the required columns
    spine_df = target_df[SPINE_COLS]

    return spine_df


def attach_parcela_var(spine_df: pd.DataFrame, parcelas_df: pd.DataFrame, columns_to_attach: list) -> pd.DataFrame:
    # Assure join keys are the same type
    spine_df['codparcela'] = spine_df['codparcela'].astype('object')
    spine_df['fecha'] = spine_df['fecha'].astype('datetime64[ns]')

    parcelas_df['codparcela'] = parcelas_df['codparcela'].astype('object')
    parcelas_df['fecha'] = parcelas_df['fecha'].astype('datetime64[ns]')

    return spine_df.merge(parcelas_df[['codparcela', 'fecha'] + columns_to_attach], how='left', on=['codparcela', 'fecha'])


def attach_meteo_var(
        spine_df: pd.DataFrame, meteo_df: pd.DataFrame, columns_to_attach: list, window_tolerance: int = 2
) -> pd.DataFrame:

    # Assure join keys are the same type
    spine_df['codparcela'] = spine_df['codparcela'].astype('object')
    spine_df['fecha'] = spine_df['fecha'].astype('datetime64[ns]')

    meteo_df['codparcela'] = meteo_df['codparcela'].astype('object')
    meteo_df['fecha'] = meteo_df['fecha'].astype('datetime64[ns]')

    # Join dataframes
    total_df = pd.merge_asof(
        spine_df.sort_values('fecha'), meteo_df[[
            'codparcela', 'fecha'] + columns_to_attach].sort_values('fecha'),
        by='codparcela', left_on='fecha', right_on='fecha',
        direction='backward', tolerance=pd.Timedelta(days=window_tolerance))

    return total_df


def calculate_climatic_stats_time_window(meteo_df: pd.DataFrame, rolling_window: str = '30D') -> pd.DataFrame:
    # Format dataframe
    stat_df = meteo_df[['codparcela', 'fecha'] + METEO_COLUMNS].copy()
    stat_df['fecha'] = pd.to_datetime(stat_df['fecha'])
    stat_df.sort_values(by=['codparcela', 'fecha'], inplace=True)
    stat_df.set_index('fecha', inplace=True)

    # Calculate descriptive statistics
    stat_df = stat_df.groupby('codparcela').rolling(rolling_window).agg(
        ['count', 'mean', 'std', 'min', 'median', 'max'])
    stat_df.columns = ['_'.join(col) + '_' + rolling_window for col in stat_df.columns]
    stat_df = stat_df.reset_index()

    return stat_df


def calculate_week_number(parcelas_df: pd.DataFrame) -> pd.DataFrame:
    df = parcelas_df.copy()
    # Convert fecha column to datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Add a new column for the week number
    df['week_number'] = df['fecha'].dt.isocalendar().week
    return df


def calculates_days_in_phenological_state_current_and_previous(parcelas_df: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the relevant columns
    df = parcelas_df.copy()

    # Sort by parcel and date
    df = df.sort_values(by=['codparcela', 'fecha'])

    # Calculate the difference in days between consecutive rows for each campaign and parcel
    df['days_spent'] = df.groupby(['codparcela', 'campaña'])['fecha'].diff().dt.days

    # Identify changes in the 'estado_mayoritario' column
    df['state_change'] = df['estado_mayoritario'] != df['estado_mayoritario'].shift(1)
    df['period_id'] = df['state_change'].cumsum()

    # Calculate cumulative days spent in the current state
    df['days_in_current_state'] = df.groupby(['codparcela', 'campaña', 'period_id'])[
        'days_spent'].cumsum()

    # Calculate total days spent in the previous state
    condition = (
        (df['state_change'] == True) &
        (df['campaña'] == df['campaña'].shift(1)) &
        (df['codparcela'] == df['codparcela'].shift(1))
    )
    df['days_in_previous_state'] = np.where(
        condition, df['days_in_current_state'].shift(1), np.nan)

    # Forward fill to propagate previous state data
    df['days_in_previous_state'] = df.groupby(['codparcela', 'campaña'])[
        'days_in_previous_state'].ffill()

    df = df.dropna(subset=['days_in_current_state', 'days_in_previous_state'])

    # Remove 'days_spent' and 'period_id' columns
    df.drop(columns=['days_spent', 'period_id'], inplace=True)

    return df


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    features_df = (
        data
        .pipe(calculate_week_number)  # Apply calculate_week_number first
        # Then apply calculates_days_in_phenological_state_current_and_previous
        .pipe(calculates_days_in_phenological_state_current_and_previous)
    )
    return features_df


def merge_and_clean(features_df: pd.DataFrame, raw_meteo_data: pd.DataFrame) -> pd.DataFrame:
    # Ensure correct data types
    features_df['codparcela'] = features_df['codparcela'].astype(str)
    features_df['fecha'] = pd.to_datetime(features_df['fecha'])
    raw_meteo_data['codparcela'] = raw_meteo_data['codparcela'].astype(str)
    raw_meteo_data['fecha'] = pd.to_datetime(raw_meteo_data['fecha'])

    # Sort dataframes by relevant keys
    features_df = features_df.sort_values(by=['codparcela', 'fecha'])
    raw_meteo_data = raw_meteo_data.sort_values(by=['codparcela', 'fecha'])

    # Merge dataframes while maintaining all rows from features_df
    merged_df = pd.merge(
        features_df,
        # Include 'indice' and 'valor'
        raw_meteo_data[['codparcela', 'fecha', 'indice', 'valor']],
        on=['codparcela', 'fecha'],
        how='left'  # Maintain all rows from features_df
    )

    # Remove rows with NaN in either the 'indice' or 'valor' columns
    merged_df_clean = merged_df.dropna(subset=['indice', 'valor'])

    return merged_df_clean


def calculate_and_merge_climatic_stats(meteo_df: pd.DataFrame, rolling_window: str = '30D') -> pd.DataFrame:
    """
    Calculate rolling climatic statistics for each unique 'codparcela' and merge them back into the original DataFrame.

    Parameters:
    - meteo_df (pd.DataFrame): The original DataFrame containing climate data with columns 'codparcela', 'fecha', 'indice', and 'valor'.
    - rolling_window (str): The time window to use for the rolling calculations (e.g., '30D' for 30 days).

    Returns:
    - pd.DataFrame: The DataFrame with added columns containing rolling statistics.
    """

    # Convert 'fecha' to a datetime object if it's not already
    meteo_df['fecha'] = pd.to_datetime(meteo_df['fecha'])

    # Pivot the DataFrame: convert 'indice' values into separate columns
    pivoted_df = meteo_df.pivot_table(
        index=['codparcela', 'fecha'], columns='indice', values='valor', aggfunc='first'
    )

    # Flatten the columns and rename them for clarity
    pivoted_df.columns = ['valor_' + str(col) for col in pivoted_df.columns]

    # Reset index to keep 'codparcela' and 'fecha' as columns for a future merge
    pivoted_df.reset_index(inplace=True)

    # Ensure that 'fecha' is not included in the numeric columns
    numeric_cols = pivoted_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'fecha' in numeric_cols:
        numeric_cols.remove('fecha')

    # Set 'fecha' as the index for rolling calculations
    pivoted_df.set_index('fecha', inplace=True)

    # Calculate rolling statistics for each numeric column
    grouped_df = pivoted_df.groupby('codparcela')[numeric_cols]
    stats_df = grouped_df.rolling(rolling_window).agg(
        ['count', 'mean', 'std', 'min', 'median', 'max']
    )

    # Flatten the MultiIndex in the columns
    stats_df.columns = ['_'.join(col).strip() for col in stats_df.columns]
    stats_df.reset_index(inplace=True)

    # Fill NaN values with zeros
    stats_df.fillna(0, inplace=True)

    # Merge calculated statistics back into the original DataFrame
    final_df = pd.merge(meteo_df, stats_df, on=['codparcela', 'fecha'], how='left')

    return final_df
