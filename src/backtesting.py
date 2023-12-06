import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def load_backtesting_data(file_path):
    """
    Load and preprocess backtesting data for trading simulation.

    :param file_path: Path to the CSV file containing backtesting data.
    :return: DataFrame with backtesting data.
    """
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    return df

def simulate_trading(df, model, scaler):
    """
    Simulate trading based on the model's predictions.

    :param df: DataFrame with backtesting data.
    :param model: Trained Keras model.
    :param scaler: Scaler object used for feature scaling.
    :return: DataFrame with trading simulation results.
    """
    features = df.drop('target', axis=1)
    scaled_features = scaler.transform(features)

    predictions = model.predict(scaled_features)
    df['predicted_movement'] = (predictions > 0.5).astype(int)

    df['position'] = df['predicted_movement'].shift(1)
    df['price_change'] = df['close'].pct_change()
    df['strategy_return'] = df['position'] * df['price_change']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

    return df

def calculate_performance_metrics(df):
    """
    Calculate performance metrics for the trading strategy.

    :param df: DataFrame with trading simulation results.
    """
    total_return = df['cumulative_return'].iloc[-1] - 1
    print(f"Total Return: {total_return:.2%}")

if __name__ == '__main__':
    backtesting_file = 'data/solana_backtesting_data.csv'
    model_file = 'models/solana_price_predictor.h5'

    try:
        df = load_backtesting_data(backtesting_file)

        scaler = StandardScaler().fit(df.drop('target', axis=1))
        model = load_model(model_file)

        df_with_trades = simulate_trading(df, model, scaler)
        calculate_performance_metrics(df_with_trades)

        df_with_trades.to_csv('data/backtesting_results.csv')
        print("Backtesting results saved to 'data/backtesting_results.csv'.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"Error in backtesting: {e}")
