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

    # Impute to 3 when target column is greater than 3
    parcelas_with_target_df.loc[parcelas_with_target_df['target'] >= 3, 'target'] = 3
    
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
    if columns_to_attach == 0:
        return spine_df
    # Assure join keys are the same type
    spine_df['codparcela'] = spine_df['codparcela'].astype('object')
    spine_df['fecha'] = spine_df['fecha'].astype('datetime64[ns]')

    parcelas_df['codparcela'] = parcelas_df['codparcela'].astype('object')
    parcelas_df['fecha'] = parcelas_df['fecha'].astype('datetime64[ns]')

    return spine_df.merge(parcelas_df[['codparcela', 'fecha'] + columns_to_attach], how='left', on=['codparcela','fecha'])


def attach_meteo_var(
        spine_df:pd.DataFrame, meteo_df:pd.DataFrame, columns_to_attach:list, window_tolerance:int=2
        )->pd.DataFrame:
    if columns_to_attach == 0:
        return spine_df
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


def create_feature_frame(
    parcelas_df: pd.DataFrame,
    meteo_df: pd.DataFrame,
    parcelas_cols_to_attach: list[str],
    meteo_cols_to_attach: list[str],
) -> pd.DataFrame:
    feature_frame = (
        parcelas_df.pipe(create_spine)
        .pipe(attach_parcela_var, parcelas_df, parcelas_cols_to_attach)
        .pipe(attach_meteo_var, meteo_df, meteo_cols_to_attach)
    )
    return feature_frame
