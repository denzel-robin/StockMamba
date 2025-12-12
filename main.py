import yfinance as yf
import numpy as np

from prepare_dataset import *
from mamba_model import *
from train_test import *

# Define the ticker symbol
ticker_symbol = "AAPL"

# Create a Ticker object
ticker = yf.Ticker(ticker_symbol)

# Fetch historical market data
historical_data = ticker.history(period="10y")  # data for the last year
historical_data = historical_data.drop(columns=['Dividends', 'Volume', 'Stock Splits'])

print("\nPreparing train/test data...")
train_dataloader, valid_dataloader, test_dataloader = PrepareDataset(historical_data, BATCH_SIZE=2)

print("\nTraining STGmamba model...")
STGmamba, STGmamba_loss = TrainSTG_Mamba(train_dataloader, valid_dataloader, num_epochs=25, mamba_features=historical_data.shape[1])
print("\nTesting STGmamba model...")
results = TestSTG_Mamba(STGmamba, test_dataloader)
