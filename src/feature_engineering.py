import pandas as pd

def calculate_moving_averages(df, windows=[5, 10, 20]):
    """
    Calculate moving averages for given window sizes.

    :param df: DataFrame containing the 'close' prices.
    :param windows: List of window sizes for moving averages.
    :return: DataFrame with moving average columns added.
    """
    for window in windows:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
    return df

def calculate_rsi(df, window=14):
    """
    Calculate the Relative Strength Index (RSI).

    :param df: DataFrame containing the 'close' prices.
    :param window: Window size for RSI calculation.
    :return: DataFrame with RSI column added.
    """
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df, window=20, num_std=2):
    """
    Calculate Bollinger Bands.

    :param df: DataFrame containing the 'close' prices.
    :param window: Window size for moving average.
    :param num_std: Number of standard deviations for the bands.
    :return: DataFrame with Bollinger Bands columns added.
    """
    df['bb_ma'] = df['close'].rolling(window=window).mean()
    df['bb_std'] = df['close'].rolling(window=window).std()
    df['bb_upper'] = df['bb_ma'] + (df['bb_std'] * num_std)
    df['bb_lower'] = df['bb_ma'] - (df['bb_std'] * num_std)
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.

    :param df: DataFrame with OHLCV data.
    :return: DataFrame with technical indicators added.
    """
    df = calculate_moving_averages(df)
    df = calculate_rsi(df)
    df = calculate_bollinger_bands(df)
    return df

if __name__ == '__main__':
    data_path = 'data/solana_training_data.csv'
    try:
        df = pd.read_csv(data_path, index_col='timestamp', parse_dates=True)
        df_with_indicators = add_technical_indicators(df)
        df_with_indicators.to_csv('data/solana_training_data_with_indicators.csv')
        print("Technical indicators added and saved to 'solana_training_data_with_indicators.csv'.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error in feature engineering: {e}")
