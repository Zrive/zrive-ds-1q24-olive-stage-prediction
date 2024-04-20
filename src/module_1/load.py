import pandas as pd
import numpy as np
import logging
import os
from typing import Optional

logging.basicConfig(level=logging.INFO)

DATA_PATH = os.path.join(os.getcwd(), "..", "..", "data")
PARCELAS_DATA_PATH = os.path.join(DATA_PATH, "muestreos_parcelas.parquet")
METEO_DATA_PATH = os.path.join(DATA_PATH, "meteo_parcelas.parquet")

PHENOLOGICAL_STATE_COLS = [f"estado_fenologico_{i}" for i in range(14, 0, -1)]


def load_raw_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading dataset from {path}")
    try:
        data = pd.read_parquet(path)
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None


def filter_parcelas_by_dates(
    df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Keeps rows that have dates between start_date and end_date (inclusive).

    REASON: Remove years with not many samples and that intersect with
    the date range in which we have meteo data.
    """
    logging.info("Filtering parcelas dataset by date")

    df["fecha"] = pd.to_datetime(df["fecha"])
    date_filter = (df["fecha"] >= start_date) & (df["fecha"] <= end_date)
    filtered_df = df[date_filter]

    logging.info(f"Dataset shape: {filtered_df.shape}")
    return filtered_df


def convert_uninformed_states_to_nan(
    df: pd.DataFrame, phenological_state_cols: list[str] = PHENOLOGICAL_STATE_COLS
) -> pd.DataFrame:
    """
    Convert phenological states with values different from 1 or 2 to NaN
    """
    for col in phenological_state_cols:
        df[col] = df[col].apply(lambda x: np.nan if x != 2.0 and x != 1.0 else x)
    return df


def remove_rows_with_all_null_phenological_states(
    df: pd.DataFrame, phenological_state_cols: list[str] = PHENOLOGICAL_STATE_COLS
) -> pd.DataFrame:
    """
    Remove rows that have all null phenological states.
    """
    logging.info("Removing rows that have all null phenological states")

    all_phenological_null_filter = df[phenological_state_cols].isnull().all(axis=1)
    filtered_df = df[~all_phenological_null_filter]

    logging.info(f"Dataset shape: {filtered_df.shape}")
    return filtered_df


def create_majority_phenological_state_column(
    df: pd.DataFrame, phenological_state_cols: list[str] = PHENOLOGICAL_STATE_COLS
) -> pd.DataFrame:
    """
    Create a column with the majority phenological state.

    If there are more than one majority state, returns the greatest one.
    Discards those rows with no majority state.

    NOTE: pass in the phenological_state_cols from biggest to smallest.
    """
    logging.info("Creating majority phenological state column")

    def get_majority_state(row) -> Optional[int]:
        for col in phenological_state_cols:
            if row[col] == 2:
                return int(col.split("_")[-1])
        return pd.NA

    df_new = df.copy()
    df_new["estado_mayoritario"] = df_new.apply(get_majority_state, axis=1)
    df_new = df_new[df_new["estado_mayoritario"].notnull()]
    df_new["estado_mayoritario"] = df_new["estado_mayoritario"].astype(int)

    logging.info(f"Dataset shape: {df_new.shape}")
    return df_new


def create_primary_key_with_codparcela_and_provincia(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a primary key column using codparcela and provincia
    """
    logging.info("create_primary_key_with_codparcela_and_provincia")

    df["pkey"] = df["codparcela"].astype(str) + "_" + df["provincia"].astype(str)

    logging.info(f"Dataset shape: {df.shape}")
    return df


def remove_low_samples_for_pkey_in_campaign(
    df: pd.DataFrame, min_num_samples: int = 10
) -> pd.DataFrame:
    """
    Remove the samples of a (codparcela, provincia) in a campaign if it has less
    than min_num_samples
    """
    logging.info("remove_codparcela_with_low_samples_in_campaign")

    parcela_counts_per_campaign = (
        df.groupby(["campaña", "pkey"])
        .size()
        .reset_index()
        .rename({0: "count_muestras_campaña"}, axis=1)
    )

    df = df.merge(
        parcela_counts_per_campaign, how="left", on=["campaña", "pkey"]
    ).sort_values(by=["pkey", "campaña"])
    df = df[df["count_muestras_campaña"] >= min_num_samples]
    df = df.drop("count_muestras_campaña", axis=1)

    logging.info(f"Dataset shape: {df.shape}")
    return df


def remove_highly_spaced_samples_for_pkey_in_campaign(
    df: pd.DataFrame, max_spacing_in_days: int = 30
) -> pd.DataFrame:
    """
    Remove the samples of a (codparcela, provincia) in a campaign if their date
    difference to the next sample is more than max_spacing_in_days
    """
    logging.info("remove_highly_spaced_samples_for_pkey_in_campaign")

    new_df = df.copy()
    new_df = new_df.sort_values(by=["pkey", "fecha"])
    new_df["diferencia_dias"] = (
        new_df.groupby(["pkey", "campaña"], as_index=False)["fecha"].diff().dt.days
    )
    new_df["diferencia_dias"] = new_df["diferencia_dias"].fillna(0)
    new_df = new_df[new_df["diferencia_dias"] <= max_spacing_in_days]
    new_df = new_df.drop("diferencia_dias", axis=1)

    logging.info(f"Dataset shape: {new_df.shape}")
    return new_df


def remove_codparcelas_not_in_meteo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eliminate codparcelas for which we do not have meteo data
    """
    logging.info("remove_codparcelas_not_in_meteo_data")

    # meteo_raw_data = load_raw_data(METEO_DATA_PATH)
    # codparcelas_in_meteo = set(meteo_raw_data["codparcela"])
    column = "codparcela"
    # codparcelas_in_meteo = set(
    #     pd.read_parquet(METEO_DATA_PATH, columns=[column])[column]
    # )
    codparcelas_in_meteo = set(load_raw_data(METEO_DATA_PATH)[column])

    df = df[df["codparcela"].isin(codparcelas_in_meteo)]

    logging.info(f"Dataset shape: {df.shape}")
    return df


def build_feature_frame(data: pd.DataFrame) -> pd.DataFrame:
    logging.info("Building feature frame")

    START_DATE = pd.Timestamp("2018-01-01")
    END_DATE = pd.Timestamp("2020-12-31")

    logging.info(f"Dataset shape before cleaning: {data.shape}")
    preprocessed_data = (
        data.pipe(filter_parcelas_by_dates, start_date=START_DATE, end_date=END_DATE)
        .pipe(remove_rows_with_all_null_phenological_states)
        .pipe(convert_uninformed_states_to_nan)
        .pipe(create_majority_phenological_state_column)
        .pipe(create_primary_key_with_codparcela_and_provincia)
        .pipe(remove_low_samples_for_pkey_in_campaign)
        .pipe(remove_highly_spaced_samples_for_pkey_in_campaign)
        .pipe(remove_codparcelas_not_in_meteo_data)
    )
    return preprocessed_data


def load_training_feature_frame() -> pd.DataFrame:
    logging.info("Loading feature frame")
    parcelas_raw_data = load_raw_data(PARCELAS_DATA_PATH)
    data = build_feature_frame(parcelas_raw_data)
    return data
