import os
import pickle
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.svm import SVR
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from models.model import Model

class SVRModel(Model):
    """
    Support Vector Regressor (SVR) Model.
    """
    def __init__(self):
        """
        Initialize the SVRModel.
        """
        load_dotenv()
        self.model: SVR = None
        self.model_path: str = os.getenv('ML_MODEL_PATH', 'saved_models/svr_model.sav')

    def load(self) -> None:
        """
        Load the SVR model from the saved file.
        """
        self.model = joblib.load(self.model_path)

    def save(self) -> None:
        """
        Save the SVR model to a file.
        """
        if self.model is not None:
            pickle.dump(self.model, open(self.model_path, 'wb'))

    def train(self, data: pd.DataFrame, labels: pd.DataFrame) -> SVR:
        """
        Train a Support Vector Regressor model.

        Args:
            data (pd.DataFrame): Features for training.
            labels (pd.DataFrame): Target for training.

        Returns:
            sklearn.svm.SVR: Trained SVR model.
        """
        model = SVR()
        model.fit(data.values, labels.values)
        self.model = model
        return model

    def predict(self, data: list) -> float:
        """
        Predict using the trained SVR model.

        Args:
            data (list): List of data for prediction.

        Returns:
            float: Predicted value.
        """
        if self.model is None:
            return None
        return float(self.model.predict(data))

    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame) -> dict:
        """
        Evaluate the performance of the SVR model.

        Args:
            data (pd.DataFrame): Features for validation.
            labels (pd.DataFrame): Target for validation.

        Returns:
            dict: Evaluation metrics (R2score, MSE, MAE).
        """
        if self.model is None:
            return {}

        predictions = self.model.predict(data)

        score = r2_score(labels, predictions)
        mse = mean_squared_error(labels, predictions)
        errors = abs(predictions - labels.values.ravel())
        mae = round(np.mean(errors), 2)

        print('R^2 Score:', score)
        print('Mean Squared Error:', mse)
        print('Mean Absolute Error:', mae)

        return {"R2score": score, "MSE": mse, "MAE": mae}
