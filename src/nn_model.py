import pandas as pd
from db_engine import DBEngine
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, History
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def build_neural_network(input_dim: int) -> Sequential:
    """
    Build a neural network model.

    Args:
        input_dim (int): Number of input dimensions.

    Returns:
        tensorflow.keras.models.Sequential: Compiled neural network model.
    """
    model = Sequential()
    model.add(Dense(128, kernel_initializer='normal', input_dim=input_dim, activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(256, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))

    model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.summary()

    return model

def train_neural_network(model: Sequential, X_train: np.ndarray,
    y_train: np.ndarray, X_val: np.ndarray, 
    y_val: np.ndarray, checkpoint_path: str) -> History:
    """
    Train a neural network model.

    Args:
        model (tensorflow.keras.models.Sequential): Compiled neural network model.
        X_train (numpy.ndarray): Features for training.
        y_train (numpy.ndarray): Target for training.
        X_val (numpy.ndarray): Features for validation.
        y_val (numpy.ndarray): Target for validation.
        checkpoint_path (str): Filepath for ModelCheckpoint to save the best model.

    Returns:
        tensorflow.keras.callbacks.History: Training history.
    """
    # Define ModelCheckpoint callback
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    # Train the neural network
    history = model.fit(
        X_train, y_train,
        epochs=1, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint]
    )

    return history

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.DataFrame):
    """
    Evaluate the performance of a regression model.

    Args:
        model: Trained neural network model.
        X_val (pd.DataFrame): Features for validation.
        y_val (pd.DataFrame): Target for validation.
    """
    predictions = model.predict(X_val)

    score = r2_score(y_val, predictions)
    print("R^2 Score for predictions:", score)
    mse = mean_squared_error(y_val, predictions)
    print(f'Mean Squared Error: {mse}')
    errors = abs(predictions - y_val.values.ravel())
    print('Mean Absolute Error:', round(np.mean(errors), 2))
    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * (errors / y_val.values.ravel())
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')

def main():
    dbeng = DBEngine()

    train_features = dbeng.fetch_data('train_features')
    val_features = dbeng.fetch_data('val_features')
    test_features = dbeng.fetch_data('test_features')

    train_target = dbeng.fetch_data('train_target')
    val_target = dbeng.fetch_data('val_target')
    test_target = dbeng.fetch_data('test_target')

    model = build_neural_network(input_dim=train_features.shape[1])

    checkpoint_path = '/app/models/nn_model.h5'
    train_neural_network(model, train_features, train_target, val_features, val_target, checkpoint_path)

    evaluate_model(model, test_features, test_target)
    model.save(checkpoint_path)

if __name__ == "__main__":
    main()
