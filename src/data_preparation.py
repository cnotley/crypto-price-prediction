import pandas as pd
import os

def load_data(file_path):
    """
    Load data from a CSV file.

    :param file_path: Path to the CSV file.
    :return: DataFrame with loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

def preprocess_data(df):
    """
    Preprocess the loaded OHLCV data.

    :param df: DataFrame with OHLCV data.
    :return: Preprocessed DataFrame.
    """
    df.fillna(method='ffill', inplace=True)

    return df

if __name__ == '__main__':
    data_folder = 'data'
    training_file = 'solana_training_data.csv'
    backtesting_file = 'solana_backtesting_data.csv'

    try:
        training_data_path = os.path.join(data_folder, training_file)
        training_data = load_data(training_data_path)
        preprocessed_training_data = preprocess_data(training_data)

        backtesting_data_path = os.path.join(data_folder, backtesting_file)
        backtesting_data = load_data(backtesting_data_path)
        preprocessed_backtesting_data = preprocess_data(backtesting_data)

        print("Data loading and preprocessing complete.")
    except Exception as e:
        print(f"Error during data preparation: {e}")
