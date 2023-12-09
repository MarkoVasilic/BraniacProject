import pandas as pd
import json
from feature_extractor import FeatureExtractor
from db_engine import DBEngine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

class DataPreprocessor:
    def __init__(self, data_file_path: str):
        """
        Initialize the DataPreprocessor.

        :param data_file_path: A string representing the file path of the data.
        """
        self.data_file_path: str = data_file_path
        self.movies_df: Optional[pd.DataFrame] = pd.read_csv(self.data_file_path)
        self.dbeng = DBEngine()

    def filter_zero_rating_movies(self):
        """
        Filter movies with a vote_average of 0 from the DataFrame.
        """
        if self.movies_df is None:
            return
        
        self.movies_df = self.movies_df[self.movies_df['vote_average'] != 0]

    def keep_selected_columns(self):
        """
        Keep only specified columns in the DataFrame.
        """
        selected_columns = ['id', 'production_companies', 
                            'production_countries', 'keywords', 'vote_average']
        
        if self.movies_df is None:
            return
        
        self.movies_df = self.movies_df[selected_columns]
    
    def json_to_list(self):
        """
        Convert JSON format to a list and perform necessary transformations.
        """
        if self.movies_df is None:
            return
        feature_names_list = ['production_companies', 'production_countries', 'keywords']
        for feature_name in feature_names_list:
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

    def remove_empty_production_data(self):
        """
        Remove movies with empty production_companies and production_countries columns.
        """
        if self.movies_df is None:
            return
        to_drop = self.movies_df[(self.movies_df['production_companies'].astype(str) == "['']") & 
                        (self.movies_df['production_countries'].astype(str) == "['']")].index

        self.movies_df = self.movies_df.drop(to_drop, axis=0)

    def create_feature_columns(self):
        """
        Create binary feature columns for 'production_companies', 'production_countries', and 'keywords' in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with binary feature columns.
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
        self.dbeng.save_dataframe(companies_df, 'production_companies')
        self.dbeng.save_dataframe(countries_df, 'production_countries')
        self.dbeng.save_dataframe(keywords_df, 'keywords')

    def process_features_and_target(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process features and target column by extracting features, scaling, and creating a separate DataFrame for the target.

        Returns:
            pd.DataFrame: Scaled features DataFrame.
            pd.DataFrame: DataFrame for the target column.
        """
        if self.movies_df is None:
            return
        
        feature_columns = ['production_companies', 'production_countries', 'keywords']
        target_column = 'vote_average'

        # Extract features
        feat_df = self.movies_df[feature_columns]

        # Scale features
        scaler = MinMaxScaler()
        feat_scaled = pd.DataFrame(scaler.fit_transform(feat_df.astype(float)))
        feat_scaled.index = feat_df.index
        feat_scaled.columns = feat_df.columns

        # Create a separate DataFrame for the target column
        target_df = pd.DataFrame()
        target_df[target_column] = self.movies_df[target_column]

        return feat_scaled, target_df
    
    def train_test_val_split(
        self,
        df_feat: pd.DataFrame,
        df_target: pd.DataFrame,
        train_frac: float,
        val_frac: float = 0.2,
        random_state: int or None = None
    ) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split features and target into training, testing, and validation sets.

        Args:
            df_feat (pd.DataFrame): DataFrame containing features.
            df_target (pd.DataFrame): DataFrame containing the target column.
            train_frac (float): Fraction of the data to include in the training set.
            val_frac (float): Fraction of the data to include in the validation set. Default is 0.2.
            random_state (int or None): Seed for reproducibility. Default is None.

        Returns:
            tuple: Tuple containing training and testing sets for features and target.
                Format: ((train_features, train_target), (val_features, val_target), (test_features, test_target))
        """
        # Splitting training set from the rest of the dataset
        train_features, temp_features, train_target, temp_target = train_test_split(
            df_feat, df_target, test_size=(1 - train_frac), random_state=random_state
        )

        # Splitting the remaining data into validation and test sets
        test_features, val_features, test_target, val_target = train_test_split(
            temp_features, temp_target, test_size=1 - val_frac, random_state=random_state
        )

        return (train_features, train_target), (val_features, val_target), (test_features, test_target)

    def save_dataframes_to_postgres(self):
        scaled_features, target_ratings = self.process_features_and_target()
        train_set, val_set, test_set = self.train_test_val_split(scaled_features, target_ratings, train_frac=0.8, val_frac=0.5, random_state=42)
        train_features, train_target = train_set
        val_features, val_target = val_set
        test_features, test_target = test_set

        # Save DataFrames to PostgreSQL tables
        self.dbeng.save_dataframe(train_features, 'train_features')
        self.dbeng.save_dataframe(val_features, 'val_features')
        self.dbeng.save_dataframe(test_features, 'test_features')

        self.dbeng.save_dataframe(train_target, 'train_target')
        self.dbeng.save_dataframe(val_target, 'val_target')
        self.dbeng.save_dataframe(test_target, 'test_target')

    def preprocess(self):
        self.filter_zero_rating_movies()
        self.keep_selected_columns()
        self.json_to_list()
        self.remove_empty_production_data()
        self.create_feature_columns()
        self.save_dataframes_to_postgres()
        