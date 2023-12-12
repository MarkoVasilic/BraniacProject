import unittest
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from data.data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):

    def setUp(self):
        
        self.data_file_path = os.path.join(os.getcwd(), '..', 'src/dataset/data.csv')
        self.preprocessor = DataPreprocessor(self.data_file_path)

    def test_filter_zero_rating_movies(self):
        self.preprocessor.filter_zero_rating_movies()
        # Assert that no movies with vote_average 0 exist in the DataFrame
        self.assertFalse(any(self.preprocessor.movies_df['vote_average'] == 0))

    def test_keep_selected_columns(self):
        self.preprocessor.filter_zero_rating_movies()
        self.preprocessor.keep_selected_columns()
        # Assert that only specified columns are present in the DataFrame
        expected_columns = ['id', 'production_companies', 'production_countries', 'keywords', 'vote_average']
        self.assertListEqual(list(self.preprocessor.movies_df.columns), expected_columns)

    def test_json_to_list(self):
        # Run json_to_list method
        self.preprocessor.filter_zero_rating_movies()
        self.preprocessor.keep_selected_columns()
        self.preprocessor.json_to_list()

        # Assertions for production_companies column
        self.assertEqual(self.preprocessor.movies_df['production_companies'].apply(lambda x: all(isinstance(i, str) for i in x)).all(), True)

        # Assertions for production_countries column
        self.assertEqual(self.preprocessor.movies_df['production_countries'].apply(lambda x: all(isinstance(i, str) for i in x)).all(), True)

        # Assertions for keywords column
        self.assertEqual(self.preprocessor.movies_df['keywords'].apply(lambda x: all(isinstance(i, str) for i in x)).all(), True)

    def test_remove_empty_production_data(self):
        # Run remove_empty_production_data method
        self.preprocessor.filter_zero_rating_movies()
        self.preprocessor.keep_selected_columns()
        self.preprocessor.json_to_list()
        self.preprocessor.remove_empty_production_data()

        # Assertions for production_companies column
        self.assertTrue(all(self.preprocessor.movies_df['production_companies'].apply(lambda x: len(x) > 0)))

        # Assertions for production_countries column
        self.assertTrue(all(self.preprocessor.movies_df['production_countries'].apply(lambda x: len(x) > 0)))

        # Assertions for keywords column
        self.assertTrue(all(self.preprocessor.movies_df['keywords'].apply(lambda x: len(x) > 0)))

if __name__ == '__main__':
    unittest.main()
