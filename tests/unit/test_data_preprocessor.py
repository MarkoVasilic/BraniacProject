import unittest
import pandas as pd
from src.data.data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        self.data_file_path = 'src/dataset/data.csv'
        self.preprocessor = DataPreprocessor(self.data_file_path)

    def test_filter_zero_rating_movies(self):
        self.preprocessor.filter_zero_rating_movies()
        # Assert that no movies with vote_average 0 exist in the DataFrame
        self.assertFalse(any(self.preprocessor.movies_df['vote_average'] == 0))

    def test_keep_selected_columns(self):
        self.preprocessor.keep_selected_columns()
        # Assert that only specified columns are present in the DataFrame
        expected_columns = ['id', 'production_companies', 'production_countries', 'keywords', 'vote_average']
        self.assertListEqual(list(self.preprocessor.movies_df.columns), expected_columns)

    def test_json_to_list(self):
        # Run json_to_list method
        self.preprocessor.json_to_list()

        # Assertions for production_companies column
        self.assertIsInstance(self.preprocessor.movies_df['production_companies'], list)
        self.assertEqual(self.preprocessor.movies_df['production_companies'].apply(lambda x: all(isinstance(i, str) for i in x)).all(), True)

        # Assertions for production_countries column
        self.assertIsInstance(self.preprocessor.movies_df['production_countries'], list)
        self.assertEqual(self.preprocessor.movies_df['production_countries'].apply(lambda x: all(isinstance(i, str) for i in x)).all(), True)

        # Assertions for keywords column
        self.assertIsInstance(self.preprocessor.movies_df['keywords'], list)
        self.assertEqual(self.preprocessor.movies_df['keywords'].apply(lambda x: all(isinstance(i, str) for i in x)).all(), True)

    def test_remove_empty_production_data(self):
        # Run remove_empty_production_data method
        self.preprocessor.remove_empty_production_data()

        # Assertions for production_companies column
        self.assertTrue(all(self.preprocessor.movies_df['production_companies'].apply(lambda x: len(x) > 0)))

        # Assertions for production_countries column
        self.assertTrue(all(self.preprocessor.movies_df['production_countries'].apply(lambda x: len(x) > 0)))

        # Assertions for keywords column
        self.assertTrue(all(self.preprocessor.movies_df['keywords'].apply(lambda x: len(x) > 0)))

    def test_create_feature_columns(self):
        result = self.preprocessor.create_feature_columns()
        # Add assertions related to the creation of binary feature columns
        self.assertIsInstance(result, dict)
        self.assertIn('production_companies', result)
        self.assertIn('production_countries', result)
        self.assertIn('keywords', result)

    def test_process_features_and_target(self):
        # Run process_features_and_target method
        features, target = self.preprocessor.process_features_and_target()

        # Assertions for features DataFrame
        self.assertIsInstance(features, pd.DataFrame)
        for column in features.columns:
            self.assertTrue(all(0 <= features[column]) and all(features[column] <= 1))

        # Assertions for target DataFrame
        self.assertIsInstance(target, pd.DataFrame)
        self.assertTrue(all(0 <= target['vote_average']) and all(target['vote_average'] <= 1))

if __name__ == '__main__':
    unittest.main()
