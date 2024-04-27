import os
import sys
import pandas as pd
import numpy as np
import unittest
from pandas.testing import assert_frame_equal

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.create_spine import generate_target, TARGET_COLS, SPINE_COLS


class TestGenerateTarget(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.data = pd.DataFrame({
            'codparcela': [1, 1, 1, 2, 2, 2],
            'fecha': pd.to_datetime(['2022-01-01', '2022-01-15', '2022-01-29', '2022-01-01', '2022-01-15','2022-02-08']),
            'estado_mayoritario': [1, 2, 3, 1, 2, 4]
        })


    def test_basic_functionality(self):
        result = generate_target(self.data, window_size=14, window_tolerance=2)
        expected_data = {
            'codparcela': [1, 1, 2],
            'fecha_futuro': pd.to_datetime(['2022-01-15', '2022-01-29', '2022-01-15']),
            'fecha_future': pd.to_datetime(['2022-01-15', '2022-01-29', '2022-01-15']),
            'estado_mayoritario_future': [2.0, 3.0, 2.0],
            'target': [1.0, 1.0, 1.0]
        }
        expected = pd.DataFrame(expected_data)
        assert_frame_equal(result[expected.columns].reset_index(drop=True), expected.reset_index(drop=True))


    def test_window_size_impact(self):
        result = generate_target(self.data, window_size=30, window_tolerance=2)
        expected_data = {
            'codparcela': [1],
            'fecha_futuro': pd.to_datetime(['2022-01-31']),
            'fecha_future': pd.to_datetime(['2022-01-29']),
            'estado_mayoritario_future': [3.0],
            'target': [2.0]
        }
        expected = pd.DataFrame(expected_data)
        assert_frame_equal(result[expected.columns].reset_index(drop=True), expected.reset_index(drop=True))


    def test_tolerance_impact(self):
        # Increase tolerance to see impact
        result = generate_target(self.data, window_size=14, window_tolerance=10)
        expected_data = {
            'codparcela': [1, 1, 2, 2],
            'fecha_futuro': pd.to_datetime(['2022-01-15', '2022-01-29', '2022-01-15', '2022-01-29']),
            'fecha_future': pd.to_datetime(['2022-01-15', '2022-01-29', '2022-01-15', '2022-02-08']),
            'estado_mayoritario_future': [2.0, 3.0, 2.0, 4.0],
            'target': [1.0, 1.0, 1.0, 2.0]
        }
        expected = pd.DataFrame(expected_data)
        print(result)
        print(expected)
        assert_frame_equal(result[expected.columns].reset_index(drop=True), expected.reset_index(drop=True))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestGenerateTarget('test_basic_functionality'))
    suite.addTest(TestGenerateTarget('test_window_size_impact'))
    suite.addTest(TestGenerateTarget('test_tolerance_impact'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
    