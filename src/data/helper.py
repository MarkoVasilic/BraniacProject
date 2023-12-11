import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
from data.feature_extractor import FeatureExtractor
from db.db_engine import DBEngine
from db.dataloader import DataLoader

def train_test_val_split(
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

def save_dataframes_to_postgres(scaled_features: pd.DataFrame, target_ratings: pd.DataFrame) -> None:
    """
    Save scaled features and target ratings to PostgreSQL tables.

    Args:
        scaled_features (pd.DataFrame): Scaled features DataFrame.
        target_ratings (pd.DataFrame): Target ratings DataFrame.
    """
    train_set, val_set, test_set = train_test_val_split(scaled_features, target_ratings, train_frac=0.8, val_frac=0.5, random_state=42)
    train_features, train_target = train_set
    val_features, val_target = val_set
    test_features, test_target = test_set

    dbeng = DBEngine()

    # Save DataFrames to PostgreSQL tables
    dbeng.save_dataframe(train_features, 'train_features')
    dbeng.save_dataframe(val_features, 'val_features')
    dbeng.save_dataframe(test_features, 'test_features')

    dbeng.save_dataframe(train_target, 'train_target')
    dbeng.save_dataframe(val_target, 'val_target')
    dbeng.save_dataframe(test_target, 'test_target')

def save_features_to_postgres(features: dict) -> None:
    """
    Save features to PostgreSQL tables.

    Args:
        features (dict): Dictionary of feature DataFrames.
    """
    dbeng = DBEngine()
    for key, value in features.items():
        dbeng.save_dataframe(value, key)

def scale_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale features using Min-Max scaling.

    Args:
        feat_df (pd.DataFrame): Input features DataFrame.

    Returns:
        pd.DataFrame: Scaled features DataFrame.
    """
    scaler = MinMaxScaler()
    scaled_features = pd.DataFrame(scaler.fit_transform(feat_df.astype(float)))
    return scaled_features

def process_data_for_predict(db_production_companies, db_production_countries, db_keywords,
                              production_companies, production_countries, keywords) -> pd.Series:
    """
    Process data for prediction.

    Args:
        db_production_companies (list): List of strings representing production companies.
        db_production_countries (list): List of strings representing production countries.
        db_keywords (list): List of strings representing keywords.
        production_companies (pd.Series): Input production companies data.
        production_countries (pd.Series): Input production countries data.
        keywords (pd.Series): Input keywords data.

    Returns:
        pd.Series: Processed and scaled features for prediction.
    """
    feat_ext = FeatureExtractor()

    feat_company = feat_ext.process_column(pd.Series(production_companies), db_production_companies).values[0]
    feat_country = feat_ext.process_column(pd.Series(production_countries), db_production_countries).values[0]
    feat_keyword = feat_ext.process_column(pd.Series(keywords), db_keywords).values[0]

    data_loader = DataLoader()
    features, _ = data_loader.load_data()
    features = features['train_features']
    new_row = [feat_company, feat_country, feat_keyword]
    features.loc[len(features)] = new_row

    features = scale_features(features)

    return features.iloc[-1]


