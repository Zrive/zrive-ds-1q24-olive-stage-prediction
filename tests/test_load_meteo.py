import pandas as pd
import numpy as np

from src.load_meteo import (
    combine_indices,
    drop_nans_for_indices,
    drop_zeros_for_indices,
    normalize_indice_values,
)


def test_combine_indices() -> None:
    df = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "FAPAR",
                "S2A_MSIL2A__resampled_gndvi",
                "S2A_MSIL2A_resampled_gndvi",
                "S2A_MSIL2A_20211002T104901_N0301_02T140006_savi",
                "S2A_MSIL2A_20210928T110841_N0301_T141544_savi",
            ]
        }
    )
    prev_cardinality = df["indice"].nunique()
    expected = pd.DataFrame(
        {"indice": ["FAPAR", "FAPAR", "GNDVI", "GNDVI", "SAVI", "SAVI"]}
    )

    result = combine_indices(df)
    new_cardinality = result["indice"].nunique()

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    assert prev_cardinality >= new_cardinality


def test_drop_nans_for_indices() -> None:
    df = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "FAPAR",
                "GNDVI",
                "GNDVI",
                "FAPAR",
                "SAVI",
                "FAPAR",
                "SAVI",
            ],
            "valor": [np.nan, 2, np.nan, 1, np.nan, np.nan, 3, 4],
        }
    )
    expected = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "GNDVI",
                "GNDVI",
                "SAVI",
                "FAPAR",
                "SAVI",
            ],
            "valor": [2, np.nan, 1, np.nan, 3, 4],
        }
    )
    indices = ["FAPAR"]
    result = drop_nans_for_indices(df, indices=indices)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_drop_zeros_for_indices() -> None:
    df = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "FAPAR",
                "GNDVI",
                "GNDVI",
                "FAPAR",
                "SAVI",
                "FAPAR",
                "SAVI",
            ],
            "valor": [0, 2, np.nan, 1, 0, np.nan, 3, 0],
        }
    )
    expected = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "GNDVI",
                "GNDVI",
                "SAVI",
                "FAPAR",
                "SAVI",
            ],
            "valor": [2, np.nan, 1, np.nan, 3, 0],
        }
    )
    indices = ["FAPAR"]
    result = drop_zeros_for_indices(df, indices=indices)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_normalize_indice_values() -> None:
    df = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "FAPAR",
                "GNDVI",
                "GNDVI",
                "FAPAR",
                "FAPAR",
                "FAPAR",
                "FAPAR",
            ],
            "valor": [255, 0, 50, 30, 0.4, 1, 3, 100],
        }
    )
    expected = pd.DataFrame(
        {
            "indice": [
                "FAPAR",
                "FAPAR",
                "GNDVI",
                "GNDVI",
                "FAPAR",
                "FAPAR",
                "FAPAR",
                "FAPAR",
            ],
            "valor": [
                1.0,
                0.0,
                50.0,
                30.0,
                0.4,
                1.0,
                0.011764705882352941,
                0.39215686274509803,
            ],
        }
    )
    indice = "FAPAR"
    max_value = 255
    result = normalize_indice_values(
        df, indice=indice, max_non_normalized_value=max_value
    )
    print(result)
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
