import unittest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.data_preprocessor import DataPreprocessor
from data.feature_extractor import FeatureExtractor

class TestIntegrated(unittest.TestCase):

    def setUp(self):
        self.data_file_path = os.path.join(os.getcwd(), '..', 'src/dataset/data.csv')
        self.preprocessor = DataPreprocessor(self.data_file_path)
        self.feature_extractor = FeatureExtractor()

    def test_integrated_preprocess(self):
        
        self.preprocessor.movies_df: pd.read_csv(self.data_file_path)

        self.preprocessor.filter_zero_rating_movies()
        self.preprocessor.keep_selected_columns()
        self.preprocessor.json_to_list()
        self.preprocessor.remove_empty_production_data()
        self.preprocessor.create_feature_columns()
        features, _ = self.preprocessor.process_features_and_target()

        # Assertions for features DataFrame
        self.assertIsInstance(features, pd.DataFrame)
        for column in features.columns:
            self.assertTrue(all(0 <= features[column]) and all(features[column] <= 1))


if __name__ == '__main__':
    unittest.main()
