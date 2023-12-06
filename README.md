Certainly! Below is a detailed, highly technical README.md file for your project. It includes an overview of the project, its structure, how to set it up, run it, and an explanation of each script.

---

# Solana Price Prediction Model

## Overview
This project is focused on predicting the price movements of the cryptocurrency Solana (SOL) using minute-by-minute candlestick data (OHLCV). It employs a deep learning approach based on the principles of Depth-µP (Depth Maximal Update Parametrization). The project covers data preparation, feature engineering, model training and evaluation, and includes a backtesting simulation to assess the strategy's performance.

## Project Structure
- `data_preparation.py`: Loads and preprocesses raw OHLCV data.
- `feature_engineering.py`: Adds technical indicators to the data for enhanced model input.
- `model.py`: Defines a deep residual neural network model based on Depth-µP.
- `training.py`: Handles model training using the prepared data.
- `evaluation.py`: Evaluates the model on a separate backtesting dataset.
- `backtesting.py`: Simulates trading using the model's predictions and calculates performance metrics.

## Setup
1. **Environment Setup**:
   - Ensure Python 3.8+ is installed.
   - Install necessary libraries: `tensorflow`, `pandas`, `scikit-learn`, etc.
2. **Data Preparation**:
   - Place your Solana OHLCV data in the `data/` directory, split into two CSV files: one for training and another for backtesting.

## Running the Project
1. **Data Preparation**:
   - Run `python data_preparation.py` to load and preprocess the data.
2. **Feature Engineering**:
   - Execute `python feature_engineering.py` to add technical indicators to the data.
3. **Model Training**:
   - Run `python training.py` to train the neural network on the prepared data.
4. **Model Evaluation**:
   - Use `python evaluation.py` to evaluate the trained model on the backtesting dataset.
5. **Backtesting Simulation**:
   - Execute `python backtesting.py` to simulate trading and analyze the strategy's performance.

## Script Details
### `data_preparation.py`
- Loads existing CSV data files.
- Preprocesses data by handling missing values and normalizing features.

### `feature_engineering.py`
- Adds technical indicators like moving averages, RSI, and Bollinger Bands to the data.
- Saves the enhanced dataset for model training.

### `model.py`
- Constructs a deep residual network using Depth-µP principles.
- Includes functions for building and scaling residual blocks.

### `training.py`
- Splits the data into training and validation sets.
- Trains the model and saves the best performing iteration.

### `evaluation.py`
- Loads the trained model and evaluates it using classification metrics.
- Reports performance on the backtesting data set.

### `backtesting.py`
- Simulates a trading strategy based on the model's predictions.
- Calculates and reports the total return and other performance metrics.

## Notes
- Ensure consistency in data scaling between training and evaluation phases.
- Modify hyperparameters in `model.py` and `training.py` as per your experimental setup.

## Author
Christopher Notley