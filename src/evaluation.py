import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def load_backtesting_data(file_path):
    """
    Load and preprocess backtesting data.

    :param file_path: Path to the CSV file containing backtesting data.
    :return: Feature and target arrays.
    """
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    X = df.drop('target', axis=1)
    y = df['target']

    return X, y

def scale_features(X, scaler):
    """
    Apply scaling to features.

    :param X: Features to be scaled.
    :param scaler: Scaler object used for scaling.
    :return: Scaled features.
    """
    return scaler.transform(X)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.

    :param model: Trained Keras model.
    :param X_test: Test features.
    :param y_test: Test target.
    """
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()

    print("Classification Report:")
    print(classification_report(y_test, y_pred_binary))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_binary))

if __name__ == '__main__':
    backtesting_file = 'data/solana_backtesting_data.csv'
    model_file = 'models/solana_price_predictor.h5'

    try:
        X_test, y_test = load_backtesting_data(backtesting_file)

        scaler = StandardScaler().fit(X_test)
        X_test_scaled = scale_features(X_test, scaler)

        model = load_model(model_file)
        evaluate_model(model, X_test_scaled, y_test)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error in model evaluation: {e}")
