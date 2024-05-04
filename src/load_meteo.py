import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

DATA_PATH = os.path.join(os.getcwd(), "data")
METEO_DATA_PATH = os.path.join(DATA_PATH, "meteo_parcelas.parquet")
METEO_DATA_PATH = os.path.join(DATA_PATH, "clean_meteo.parquet")


INDICES_TO_DROP_NANS = ["SSM"]
INDICES_TO_DROP_ZEROS = ["NDVI", "NDWI", "SAVI", "GNDVI", "SIPI"]

INDICE_TO_NORMALIZE = "FAPAR"
MAX_NON_NORMALIZED_FAPAR_VALUE = 255.0


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load dataset from a parquet file.
    """
    logging.info(f"Loading dataset from {path}")
    try:
        data = pd.read_parquet(path)
        return data
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return None


def combine_indices(df: pd.DataFrame) -> pd.DataFrame:
    """
    In the "indice" feature, there are some categories that are unnecesarily
    splitted into several more by date.

    Combine indices into a single category by extracting the last part of the string
    after splitting by '_'.
    """
    logging.info("Executing combine_indices")

    df["indice"] = df["indice"].apply(lambda x: x.split("_")[-1].upper())

    logging.info(f"Dataset shape after operation: {df.shape}")
    return df


def drop_nans_for_indices(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    """
    Remove rows with NaN values in the 'valor' column for specific indices.
    """
    logging.info(f"Executing drop_nans_for_indice {indices}")

    indice_filter = df["indice"].isin(indices)
    isnan_filter = df["valor"].isnull()
    df = df.drop(df[indice_filter & isnan_filter].index)

    logging.info(f"Dataset shape after operation: {df.shape}")
    return df


def drop_zeros_for_indices(df: pd.DataFrame, indices: list[str]) -> pd.DataFrame:
    """
    Remove rows with a value of 0.0 in the 'valor' column for specific indices.
    """
    logging.info(f"Executing drop_zeros_for_indices {indices}")

    indice_filter = df["indice"].isin(indices)
    iszero_filter = df["valor"] == 0.0
    df = df.drop(df[indice_filter & iszero_filter].index)

    logging.info(f"Dataset shape after operation: {df.shape}")
    return df


def normalize_indice_values(
    df: pd.DataFrame, indice: str, max_non_normalized_value: float
) -> pd.DataFrame:
    """
    Normalize values in the 'valor' column that satisfy a certain condition
    for a specific 'indice' by dividing them by max_non_normalized_value.
    """
    logging.info(
        f"Executing normalize_indice_values {indice},"
        + f" dividing by {max_non_normalized_value}"
    )
    indice_filter = df["indice"] == indice
    not_normalized_filter = df["valor"] > 1.0

    df.loc[indice_filter & not_normalized_filter, "valor"] /= max_non_normalized_value

    logging.info(f"Dataset shape after operation: {df.shape}")
    return df


def clean_meteo_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform cleaning operations on meteorological data.
    """
    logging.info("Executing clean_meteo_data")

    logging.info(f"Initial dataset shape: {data.shape}")
    preprocessed_data = (
        data.pipe(drop_nans_for_indices, indices=INDICES_TO_DROP_NANS)
        .pipe(combine_indices)
        .pipe(drop_zeros_for_indices, indices=INDICES_TO_DROP_ZEROS)
        .pipe(
            normalize_indice_values,
            indice=INDICE_TO_NORMALIZE,
            max_non_normalized_value=MAX_NON_NORMALIZED_FAPAR_VALUE,
        )
        
    )
    return preprocessed_data


def load_clean_meteo_data() -> pd.DataFrame:
    """
    Load and clean meteorological data.
    """
    logging.info("Loading clean meteo dataset")
    meteo_raw_data = load_raw_data(METEO_DATA_PATH)
    data = clean_meteo_data(meteo_raw_data)

    data = data.pivot_table(index=['fecha', 'codparcela', 'lat', 'lon'], columns='indice', values='valor', aggfunc='mean').reset_index()

    return data
