import pandas as pd
from typing import Dict, Tuple
from db.db_engine import DBEngine

class DataLoader:
    def __init__(self):
        """
        Initialize the DataLoader.
        """
        self.train_features: pd.DataFrame = None
        self.train_labels: pd.DataFrame = None
        self.val_features: pd.DataFrame = None
        self.val_labels: pd.DataFrame = None
        self.test_features: pd.DataFrame = None
        self.test_labels: pd.DataFrame = None
    
    def load_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Load training, validation, and test data from a database using DBEngine.

        Returns:
        Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: A tuple containing dictionaries
            with keys for 'train_features', 'val_features', 'test_features' and
            'train_labels', 'val_labels', 'test_labels' respectively, and values as the
            corresponding data.
        """
        dbeng = DBEngine()

        self.train_features = dbeng.fetch_data('train_features')
        self.val_features = dbeng.fetch_data('val_features')
        self.test_features = dbeng.fetch_data('test_features')

        self.train_labels = dbeng.fetch_data('train_target')
        self.val_labels = dbeng.fetch_data('val_target')
        self.test_labels = dbeng.fetch_data('test_target')

        features = {'train_features': self.train_features, 'val_features': self.val_features, 'test_features': self.test_features}
        labels = {'train_labels': self.train_labels, 'val_labels': self.val_labels, 'test_labels': self.test_labels}
        return features, labels
