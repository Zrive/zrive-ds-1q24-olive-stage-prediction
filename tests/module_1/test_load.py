import pandas as pd
import numpy as np
from unittest.mock import patch

from src.module_1.load import (
    filter_parcelas_by_dates,
    convert_uninformed_states_to_nan,
    remove_rows_with_all_null_phenological_states,
    create_majority_phenological_state_column,
    remove_codparcela_with_multiple_provincia,
    remove_highly_spaced_samples_for_codparcela_in_campaign,
    remove_codparcelas_not_in_meteo_data,
)


def test_filter_parcelas_by_dates() -> None:
    start_date = "2015-03-15"
    end_date = "2019-04-30"

    df = pd.DataFrame(
        {
            "fecha": [
                "2014-01-01",
                "2015-03-15",
                "2016-06-30",
                "2017-09-10",
                "2018-12-25",
                "2019-04-20",
                "2020-07-05",
                "2020-10-15",
            ]
        }
    )

    expected = pd.DataFrame(
        {
            "fecha": [
                "2015-03-15",
                "2016-06-30",
                "2017-09-10",
                "2018-12-25",
                "2019-04-20",
            ]
        }
    )
    expected["fecha"] = pd.to_datetime(expected["fecha"])

    result = filter_parcelas_by_dates(df, start_date=start_date, end_date=end_date)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    assert len(result) <= len(df)
    assert result.columns.tolist() == df.columns.tolist()
    assert result["fecha"].dtype == "datetime64[ns]"


def test_convert_uninformed_states_to_nan() -> None:
    PHENOLOGICAL_STATE_COLS = [f"estado_fenologico_{i}" for i in range(14, 0, -1)]
    df = pd.DataFrame(
        {
            state: [0.5, 1.5, 2.0, 2.5, 3.0, 1.0, 100.0]
            for state in PHENOLOGICAL_STATE_COLS
        }
    )

    expected = pd.DataFrame(
        {
            state: [np.nan, np.nan, 2.0, np.nan, np.nan, 1.0, np.nan]
            for state in PHENOLOGICAL_STATE_COLS
        }
    )

    result = convert_uninformed_states_to_nan(df)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    assert expected.shape == result.shape
    assert ((result != 1.0) & (result != 2.0) & (~result.isna())).sum().sum() == 0
    assert result.dtypes.tolist() == df.dtypes.tolist()


def test_remove_rows_with_all_null_phenological_states() -> None:
    PHENOLOGICAL_STATE_COLS = [
        "estado_fenologico_1",
        "estado_fenologico_2",
        "estado_fenologico_3",
    ]

    df = pd.DataFrame(
        {
            "estado_fenologico_1": [np.nan, np.nan, np.nan],
            "estado_fenologico_2": [1.0, np.nan, np.nan],
            "estado_fenologico_3": [1.0, np.nan, 2.0],
        }
    )

    expected = pd.DataFrame(
        {
            "estado_fenologico_1": [np.nan, np.nan],
            "estado_fenologico_2": [1.0, np.nan],
            "estado_fenologico_3": [1.0, 2.0],
        }
    )

    result = remove_rows_with_all_null_phenological_states(df, PHENOLOGICAL_STATE_COLS)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )
    assert len(result) <= len(df)


def test_create_majority_phenological_state_column() -> None:
    PHENOLOGICAL_STATE_COLS = [
        "estado_fenologico_3",
        "estado_fenologico_2",
        "estado_fenologico_1",
    ]

    df = pd.DataFrame(
        {
            "estado_fenologico_1": [np.nan, np.nan, np.nan, np.nan, 2.0],
            "estado_fenologico_2": [1.0, np.nan, np.nan, 2.0, 2.0],
            "estado_fenologico_3": [1.0, np.nan, 2.0, 2.0, 1.0],
        }
    )

    expected = pd.DataFrame(
        {
            "estado_fenologico_1": [np.nan, np.nan, 2.0],
            "estado_fenologico_2": [np.nan, 2.0, 2.0],
            "estado_fenologico_3": [2.0, 2.0, 1.0],
            "estado_mayoritario": [3, 3, 2],
        }
    )

    result = create_majority_phenological_state_column(df, PHENOLOGICAL_STATE_COLS)

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_remove_codparcela_with_multiple_provincia() -> None:
    df = pd.DataFrame({"codparcela": [1, 2, 3, 3], "provincia": ["A", "B", "C", "A"]})

    result = remove_codparcela_with_multiple_provincia(df)

    expected = pd.DataFrame({"codparcela": [1, 2], "provincia": ["A", "B"]})
    pd.testing.assert_frame_equal(
        result.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_remove_highly_spaced_samples_for_codparcela_in_campaign() -> None:
    df = pd.DataFrame(
        {
            "campaña": ["A", "A", "A", "A", "A", "A", "B", "B", "B"],
            "codparcela": ["1", "1", "1", "1", "1", "1", "2", "2", "2"],
            "fecha": [
                "2022-01-01",
                "2022-01-05",
                "2022-01-10",
                "2022-01-15",
                "2022-01-20",
                "2022-01-25",
                "2022-01-01",
                "2022-01-20",
                "2022-02-25",
            ],
        }
    )
    df["fecha"] = pd.to_datetime(df["fecha"])

    expected = pd.DataFrame(
        {
            "campaña": ["A", "A", "A", "A", "A", "A", "B", "B"],
            "codparcela": ["1", "1", "1", "1", "1", "1", "2", "2"],
            "fecha": [
                "2022-01-01",
                "2022-01-05",
                "2022-01-10",
                "2022-01-15",
                "2022-01-20",
                "2022-01-25",
                "2022-01-01",
                "2022-01-20",
            ],
        }
    )
    expected["fecha"] = pd.to_datetime(expected["fecha"])

    result = remove_highly_spaced_samples_for_codparcela_in_campaign(
        df, max_spacing_in_days=30
    )

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )


@patch("src.module_1.load.load_raw_data")
def test_remove_codparcelas_not_in_meteo_data(mock_load_raw_data):
    mock_load_raw_data.return_value = pd.DataFrame({"codparcela": [0, 1, 2, 3]})

    df = pd.DataFrame({"codparcela": [1, 2, 3, 4, 5]})

    result = remove_codparcelas_not_in_meteo_data(df)

    expected = pd.DataFrame({"codparcela": [1, 2, 3]})

    pd.testing.assert_frame_equal(
        result.reset_index(drop=True), expected.reset_index(drop=True)
    )
