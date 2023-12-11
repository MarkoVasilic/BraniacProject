from typing import Tuple, Optional, Dict
import pandas as pd
import json
from data.feature_extractor import FeatureExtractor
import data.helper as helper

class DataPreprocessor:
    def __init__(self, data_file_path: str):
        """
        Initialize the DataPreprocessor.

        Args:
            data_file_path (str): File path of the data.
        """
        self.data_file_path: str = data_file_path
        self.movies_df: Optional[pd.DataFrame] = pd.read_csv(self.data_file_path)
        self.feature_names_list = ['production_companies', 'production_countries', 'keywords']
        self.target_column = 'vote_average'

    def filter_zero_rating_movies(self) -> None:
        """
        Filter movies with a vote_average of 0 from the DataFrame.
        """
        if self.movies_df is None:
            return
        
        self.movies_df = self.movies_df[self.movies_df[self.target_column] != 0]

    def keep_selected_columns(self) -> None:
        """
        Keep only specified columns in the DataFrame.
        """
        selected_columns = ['id', 'production_companies', 'production_countries', 'keywords', 'vote_average']
        
        if self.movies_df is None:
            return
        
        self.movies_df = self.movies_df[selected_columns]
    
    def json_to_list(self) -> None:
        """
        Convert JSON format to a list and perform necessary transformations.
        """
        if self.movies_df is None:
            return
        for feature_name in self.feature_names_list:
            # STEP 1: Convert JSON format to a list
            self.movies_df[feature_name] = self.movies_df[feature_name].apply(json.loads)         
            for index, column in zip(self.movies_df.index, self.movies_df[feature_name]):
                new_feature_list = []
                for i in range(len(column)):
                    new_feature_list.append((column[i]['name']))
                self.movies_df.loc[index, feature_name]= str(new_feature_list)
            
            # STEP 2: Clean up and transform into an unsorted list
            self.movies_df[feature_name] = self.movies_df[feature_name].str.strip('[]').str.replace(' ','').str.replace("'",'')
            self.movies_df[feature_name] = self.movies_df[feature_name].str.split(',')
            
            # STEP 3: Sort list elements
            for column, index in zip(self.movies_df[feature_name], self.movies_df.index):
                new_feature_list = column
                new_feature_list.sort()
                self.movies_df.loc[index, feature_name]=str(new_feature_list)
            self.movies_df[feature_name] = self.movies_df[feature_name].str.strip('[]').str.replace(' ','').str.replace("'",'')
            
            # Set to None if the list is empty
            check = self.movies_df[feature_name].str.split(',')
            if len(check) == 0:
                self.movies_df[feature_name] = None
            else:
                self.movies_df[feature_name] = self.movies_df[feature_name].str.split(',')

    def remove_empty_production_data(self) -> None:
        """
        Remove movies with empty production_companies column or production_countries column or keywords column.
        """
        if self.movies_df is None:
            return
        to_drop = self.movies_df[
            (self.movies_df['production_companies'].astype(str) == "['']") | 
            (self.movies_df['production_countries'].astype(str) == "['']") |
            (self.movies_df['keywords'].astype(str) == "['']")
        ].index
        self.movies_df = self.movies_df.drop(to_drop, axis=0)

    def create_feature_columns(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Create binary feature columns for 'production_companies', 'production_countries', and 'keywords' in the DataFrame.

        Returns:
            dict: Dictionary containing DataFrames with binary feature columns.
        """
        if self.movies_df is None:
            return
        feature_extractor = FeatureExtractor()
        prod_companies_list = feature_extractor.generate_feature_list(self.movies_df, 'production_companies')
        prod_countries_list = feature_extractor.generate_feature_list(self.movies_df, 'production_countries')
        keywords_list = feature_extractor.generate_feature_list(self.movies_df, 'keywords')

        self.movies_df['production_companies'] = feature_extractor.process_column(self.movies_df['production_companies'], prod_companies_list)
        self.movies_df['production_countries'] = feature_extractor.process_column(self.movies_df['production_countries'], prod_countries_list)
        self.movies_df['keywords'] = feature_extractor.process_column(self.movies_df['keywords'], keywords_list)

        companies_df = pd.DataFrame(prod_companies_list, columns=['production_companies'])
        countries_df = pd.DataFrame(prod_countries_list, columns=['production_countries'])
        keywords_df = pd.DataFrame(keywords_list, columns=['keywords'])

        return {'production_companies': companies_df, 'production_countries': countries_df, 'keywords': keywords_df}
        

    def process_features_and_target(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process features and target column by extracting features, scaling, and creating a separate DataFrame for the target.

        Returns:
            pd.DataFrame: Scaled features DataFrame.
            pd.DataFrame: DataFrame for the target column.
        """
        if self.movies_df is None:
            return

        # Extract features
        feat_df = self.movies_df[self.feature_names_list]

        # Scale features
        feat_scaled = helper.scale_features(feat_df)
        feat_scaled.index = feat_df.index
        feat_scaled.columns = feat_df.columns

        # Create a separate DataFrame for the target column
        target_df = pd.DataFrame()
        target_df[self.target_column] = self.movies_df[self.target_column]

        return feat_scaled, target_df

    def preprocess(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Perform data preprocessing steps.

        Returns:
            dict: Dictionary containing DataFrames with binary feature columns.
        """
        self.filter_zero_rating_movies()
        self.keep_selected_columns()
        self.json_to_list()
        self.remove_empty_production_data()
        return self.create_feature_columns()
