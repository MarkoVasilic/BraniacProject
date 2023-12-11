import os
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from models.model import Model

class NNModel(Model):
    """
    Neural Network Model.
    """
    def __init__(self):
        """
        Initialize the NNModel.
        """
        load_dotenv()
        self.model: tf.keras.models.Sequential = None
        self.model_path: str = os.getenv('NN_MODEL_PATH', 'saved_models/nn_model.h5')

    def load(self) -> None:
        """
        Load the neural network model from the saved file.
        """
        self.model = tf.keras.models.load_model(self.model_path)

    def save(self) -> None:
        """
        Save the neural network model to a file.
        """
        if self.model is not None:
            self.model.save(self.model_path)

    def build_neural_network(self, input_dim: int) -> None:
        """
        Build a neural network model.

        Args:
            input_dim (int): Number of input dimensions.
        """
        self.model = Sequential()
        self.model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
        self.model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        self.model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        optimizer = Adam(learning_rate=0.001)

        self.model.compile(loss='mean_absolute_error', optimizer=optimizer, metrics=['mean_absolute_error'])
        self.model.summary()

    def train(self, data: dict, labels: dict) -> None:
        """
        Train a neural network model.

        Args:
            data (dict): Features for training.
            labels (dict): Target for training.
        """
        self.model.fit(
            data['train_features'], labels['train_labels'],
            epochs=500, batch_size=32,
            validation_data=(data['val_features'], labels['val_labels'])
        )
    def predict(self, data: list) -> float:
        """
        Predict using the trained neural network model.

        Args:
            data (list): List of data for prediction.

        Returns:
            float: Predicted value.
        """
        print(data)
        if self.model is None:
            return None
        return float(self.model.predict(data))

    def evaluate(self, data: pd.DataFrame, labels: pd.DataFrame) -> dict:
        """
        Evaluate the performance of the neural network model.

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
