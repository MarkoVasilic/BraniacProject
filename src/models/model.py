from abc import ABC, abstractmethod
from typing import Any

class Model(ABC):
    @abstractmethod
    def load(self, model_path: str) -> None:
        """
        Load a pre-trained model from a specified file.

        :param model_path: The path to the saved model.
        """
        pass

    @abstractmethod
    def save(self, model_path: str) -> None:
        """
        Save the current model to a specified file.

        :param model_path: The path to save the model.
        """
        pass

    @abstractmethod
    def train(self, data: Any, labels: Any) -> None:
        """
        Train the model on the provided data and labels.

        :param data: The training data.
        :param labels: The corresponding labels.
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        Make predictions using the trained model on the given data.

        :param data: The input data for predictions.
        :return: The model predictions.
        """
        pass

    @abstractmethod
    def evaluate(self, data: Any, labels: Any) -> dict:
        """
        Evaluate the performance of the model on the provided data and labels.

        :param data: The evaluation data.
        :param labels: The corresponding labels for evaluation.
        :return: A dictionary containing evaluation metrics.
        """
        pass
