import datetime

import numpy as np
import pytz as pytz
import torch
import yfinance as yf
from matplotlib import pyplot as plt
import pandas as pd

from model.trainer import evaluate_model, train_model, predict

# Get the closing price of the S&P 500 Index ETF
# spy = yf.Ticker("SPY")
# price_series = spy.history(period='max')['Close']
#
# # Calculate the 5-day rolling volatility
# volatility_series = price_series.pct_change().rolling(5).std().dropna()
# volatility_series.to_pickle('volatility_series.pkl')

volatility_series = pd.read_pickle('volatility_series.pkl')

# Show that volatility's distribution is heavily skewed to the right
volatility_series.plot.hist(bins=30)
plt.show()

# Take the log of the volatility so the distribution looks much more Gaussian
volatility_series = pd.Series(np.log(volatility_series.values), index=volatility_series.index).dropna()
volatility_series.plot.hist(bins=30)
plt.show()


# Segment training dataset using cutoff date
training_cutoff = datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone('America/New_York'))

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Store all training data in the GPU
else:
    torch.set_default_tensor_type('torch.FloatTensor')

training_series = volatility_series.loc[volatility_series.index < training_cutoff]
results = train_model(training_series, epochs=1)

predict_df = predict(results, volatility_series)

print(spy.info)