import pandas as pd
import numpy as np

TARGET_COLS = ['codparcela','fecha','estado_mayoritario']
SPINE_COLS = ['codparcela','fecha','target']
METEO_COLUMNS = ['FAPAR','GNDVI','LST','NDVI','NDWI','SAVI','SIPI','SSM']


def generate_target(parcelas_df:pd.DataFrame, window_size:int=14, window_tolerance:int=2)->pd.DataFrame:
    df = parcelas_df[TARGET_COLS].copy()
    df['fecha'] = df['fecha'].astype('datetime64[ns]') # Establish units in datetime

    # Generate date to search PHENOLOGICAL_STATE in the future
    parcelas_df['fecha_futuro'] = parcelas_df['fecha'] + pd.Timedelta(days=window_size)

    # Join between the 2 dfs with a time delta (±2 days)
    parcelas_with_target_df = pd.merge_asof(
                parcelas_df.sort_values('fecha'), df.sort_values('fecha'), 
                by='codparcela', left_on='fecha_futuro', right_on='fecha', 
                       suffixes=('', '_future'), direction='nearest', tolerance=pd.Timedelta(days=window_tolerance))

    # Generate target column - Number of phenological states that passed in the chosen window size
    parcelas_with_target_df['target'] = parcelas_with_target_df['estado_mayoritario_future'] - parcelas_with_target_df['estado_mayoritario']

    # Drop all rows with NULL in target
    parcelas_with_target_df = parcelas_with_target_df[parcelas_with_target_df['target'].notnull()]

    # Input to 0 when target column is negative
    parcelas_with_target_df.loc[parcelas_with_target_df['target'] < 0, 'target'] = 0
    
    return parcelas_with_target_df.sort_values(by=['codparcela','fecha'])


def create_spine(parcelas_df:pd.DataFrame)->pd.DataFrame:

    target_df = generate_target(parcelas_df)
    
    # Check if the necessary columns are present
    missing_columns = [col for col in SPINE_COLS if col not in target_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

    # Create the spine DataFrame by selecting the required columns
    spine_df = target_df[SPINE_COLS]

    return spine_df


def attach_parcela_var(spine_df:pd.DataFrame, parcelas_df:pd.DataFrame, columns_to_attach:list)->pd.DataFrame:
    # Assure join keys are the same type
    spine_df['codparcela'] = spine_df['codparcela'].astype('object')
    spine_df['fecha'] = spine_df['fecha'].astype('datetime64[ns]')

    parcelas_df['codparcela'] = parcelas_df['codparcela'].astype('object')
    parcelas_df['fecha'] = parcelas_df['fecha'].astype('datetime64[ns]')

    return spine_df.merge(parcelas_df[['codparcela', 'fecha'] + columns_to_attach], how='left', on=['codparcela','fecha'])


def attach_meteo_var(
        spine_df:pd.DataFrame, meteo_df:pd.DataFrame, columns_to_attach:list, window_tolerance:int=2
        )->pd.DataFrame:
    
    # Assure join keys are the same type
    spine_df['codparcela'] = spine_df['codparcela'].astype('object')
    spine_df['fecha'] = spine_df['fecha'].astype('datetime64[ns]')

    meteo_df['codparcela'] = meteo_df['codparcela'].astype('object')
    meteo_df['fecha'] = meteo_df['fecha'].astype('datetime64[ns]')

    # Join dataframes
    total_df = pd.merge_asof(
            spine_df.sort_values('fecha'), meteo_df[['codparcela', 'fecha'] + columns_to_attach].sort_values('fecha'), 
            by='codparcela', left_on='fecha', right_on='fecha', 
            direction='backward', tolerance=pd.Timedelta(days=window_tolerance))

    return total_df


def calculate_climatic_stats_time_window(meteo_df:pd.DataFrame, rolling_window:str='30D')->pd.DataFrame:
    # Format dataframe
    stat_df = meteo_df[['codparcela','fecha'] + METEO_COLUMNS].copy()
    stat_df['fecha'] = pd.to_datetime(stat_df['fecha'])
    stat_df.sort_values(by=['codparcela', 'fecha'], inplace=True)
    stat_df.set_index('fecha', inplace=True)

    # Calculate descriptive statistics
    stat_df = stat_df.groupby('codparcela').rolling(rolling_window).agg(['count','mean','std','min','median','max'])
    stat_df.columns = ['_'.join(col) + '_' + rolling_window  for col in stat_df.columns]
    stat_df = stat_df.reset_index()

    return stat_df


def calculate_week_number(parcelas_df:pd.DataFrame)->pd.DataFrame:
    df = parcelas_df.copy()
    # Convert fecha column to datetime
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Add a new column for the week number
    df['week_number'] = df['fecha'].dt.isocalendar().week
    return df


def calculates_days_in_phenological_state_current_and_previous(parcelas_df:pd.DataFrame)->pd.DataFrame:
    df = parcelas_df[['fecha','codparcela','campaña','estado_mayoritario']]

    df = df.sort_values(by=['codparcela', 'fecha'])

    # Calculate days difference
    df['days_spent'] = df.groupby(['codparcela','campaña'])['fecha'].diff().dt.days

    # Identify changes in 'estado_mayoritario' and assign a group ID for each period of the same state
    df['state_change'] = df['estado_mayoritario'].ne(df['estado_mayoritario'].shift(1))
    df['period_id'] = df['state_change'].cumsum()

    # Calculate total days spent in each state before change
    df['days_in_current_state'] = df.groupby(['codparcela','campaña','period_id'])['days_spent'].cumsum()    

    # Calculate total days spent in previous state
    df['days_in_previous_state'] = np.where(df['state_change'] == True & df['campaña'].eq(df['campaña'].shift(1)) & df['codparcela'].eq(df['codparcela'].shift(1)),
                                             df['days_in_current_state'].shift(1), np.nan)
    df['days_in_previous_state'] = df.groupby(['codparcela','campaña'])['days_in_previous_state'].ffill()
    
    return df[['codparcela','fecha','days_in_current_state','days_in_previous_state']]