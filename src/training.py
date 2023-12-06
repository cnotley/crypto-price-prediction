import pandas as pd
import tensorflow as tf
from model import build_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_training_data(file_path):
    """
    Load and preprocess training data.

    :param file_path: Path to the CSV file containing training data.
    :return: Feature and target arrays.
    """
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

    X = df.drop('target', axis=1)
    y = df['target']

    return X, y

def scale_features(X_train, X_val):
    """
    Scale features using standard scaling.

    :param X_train: Training features.
    :param X_val: Validation features.
    :return: Scaled training and validation features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    return X_train_scaled, X_val_scaled

def train_model(X_train, y_train, X_val, y_val, input_shape, depth, units, model_path):
    """
    Train the neural network model.

    :param X_train: Training features.
    :param y_train: Training target.
    :param X_val: Validation features.
    :param y_val: Validation target.
    :param input_shape: Shape of the input data.
    :param depth: Depth of the neural network.
    :param units: Units in each layer of the network.
    :param model_path: Path to save the trained model.
    """
    model = build_model(input_shape, depth, units)

    checkpoint = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32,
              callbacks=[checkpoint, early_stopping])

if __name__ == '__main__':
    data_path = 'data/solana_training_data_with_indicators.csv'
    model_path = 'models/solana_price_predictor.h5'

    try:
        X, y = load_training_data(data_path)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled, X_val_scaled = scale_features(X_train, X_val)

        input_shape = X_train_scaled.shape[1]
        depth = 10
        units = 64

        train_model(X_train_scaled, y_train, X_val_scaled, y_val, input_shape, depth, units, model_path)
        print(f"Model trained and saved to {model_path}")
    except Exception as e:
        print(f"Error in model training: {e}")
