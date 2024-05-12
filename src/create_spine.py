import pandas as pd


TARGET_COLS = ['codparcela', 'fecha', 'estado_mayoritario']
SPINE_COLS = ['codparcela', 'fecha', 'target', 'campaña', 'estado_mayoritario']


def generate_target(parcelas_df: pd.DataFrame, window_size: int = 14, window_tolerance: int = 2) -> pd.DataFrame:
    df = parcelas_df[TARGET_COLS].copy()
    df['fecha'] = df['fecha'].astype('datetime64[ns]')  # Establish units in datetime

    # Generate date to search PHENOLOGICAL_STATE in the future
    parcelas_df['fecha_futuro'] = parcelas_df['fecha'] + pd.Timedelta(days=window_size)

    # Join between the 2 dfs with a time delta (±2 days)
    parcelas_with_target_df = pd.merge_asof(
        parcelas_df.sort_values('fecha'), df.sort_values('fecha'),
        by='codparcela', left_on='fecha_futuro', right_on='fecha',
        suffixes=('', '_future'), direction='nearest', tolerance=pd.Timedelta(days=window_tolerance))

    # Generate target column - Number of phenological states that passed in the chosen window size
    parcelas_with_target_df['target'] = parcelas_with_target_df['estado_mayoritario_future'] - \
        parcelas_with_target_df['estado_mayoritario']

    # Drop all rows with NULL in target
    parcelas_with_target_df = parcelas_with_target_df[parcelas_with_target_df['target'].notnull(
    )]

    # Input to 0 when target column is negative
    parcelas_with_target_df.loc[parcelas_with_target_df['target'] < 0, 'target'] = 0

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
