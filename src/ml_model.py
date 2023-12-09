import pandas as pd
import numpy as np
import sklearn
from db_engine import DBEngine
from sqlalchemy import create_engine
from sklearn.svm import SVR
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def train_svr_model(X_train: pd.DataFrame, y_train: pd.DataFrame) -> sklearn.svm.SVR:
    """
    Train a Support Vector Regressor model.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.DataFrame): Target for training.

    Returns:
        sklearn.svm.SVR: Trained SVR model.
    """
    model = SVR()
    model.fit(X_train.values, y_train.values)
    return model

def evaluate_model(model: sklearn.svm.SVR, X_val: pd.DataFrame, y_val: pd.DataFrame):
    """
    Evaluate the performance of a regression model.

    Args:
        model: Trained regression model.
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
    test_features = dbeng.fetch_data('test_features')

    train_target = dbeng.fetch_data('train_target')
    test_target = dbeng.fetch_data('test_target')

    model = train_svr_model(train_features, train_target)

    evaluate_model(model, test_features, test_target)

    filename = '/app/models/svr_model.sav'
    pickle.dump(model, open(filename, 'wb'))

if __name__ == "__main__":
    main()
