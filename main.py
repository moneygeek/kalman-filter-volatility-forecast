import datetime

import numpy as np
import pandas as pd
import pytz as pytz
import yfinance as yf
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

from model.trainer import train_model, predict

if __name__ == "__main__":
    # Get the closing price of the S&P 500 Index ETF
    spy = yf.Ticker("SPY")
    price_series = spy.history(period='max')['Close']

    # Calculate the 5-day rolling volatility
    volatility_series = price_series.pct_change().rolling('7D').std().dropna()
    weekly_index = pd.date_range(volatility_series.index[0], volatility_series.index[-1], freq='W-FRI')
    volatility_series = volatility_series.reindex(weekly_index, method='ffill')

    # Show that volatility's distribution is heavily skewed to the right
    volatility_series.plot.hist(bins=30)
    plt.show()

    # Take the log of the volatility so the distribution looks much more Gaussian
    volatility_series = pd.Series(np.log(volatility_series.values), index=volatility_series.index).dropna()
    volatility_series.plot.hist(bins=30)
    plt.show()

    # Segment training dataset using cutoff date
    training_cutoff = datetime.datetime(2020, 1, 1, tzinfo=pytz.timezone('America/New_York'))
    training_series = volatility_series.loc[volatility_series.index < training_cutoff]

    trained_model = train_model(training_series, epochs=300)

    predict_series = predict(trained_model, volatility_series)

    # Show the performance of naive method that use previous week's volatility as the forecast
    naive_results_df = volatility_series.to_frame('Actual').join(volatility_series.shift(1).to_frame('Forecast'))
    naive_results_df = naive_results_df.loc[naive_results_df.index >= training_cutoff]

    naive_results_df.plot.line()
    naive_results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()

    print(f"[Naive] R Squared: {r2_score(naive_results_df['Actual'], naive_results_df['Forecast']):.4f}, "
          f"Mean Absolute Error: {mean_absolute_error(naive_results_df['Actual'], naive_results_df['Forecast']):.4f}")

    # Show performance of the model in log space
    log_results_df = volatility_series.to_frame('Actual').join(predict_series.shift(1).to_frame('Forecast'))
    log_results_df = log_results_df.loc[log_results_df.index >= training_cutoff]

    log_results_df.plot.line()
    log_results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()

    print(f"[Log Space] R Squared: {r2_score(log_results_df['Actual'], log_results_df['Forecast']):.4f}, "
          f"Mean Absolute Error: {mean_absolute_error(log_results_df['Actual'], log_results_df['Forecast']):.4f}")

    # Show the performance of the model in original space
    results_df = pd.DataFrame(np.exp(log_results_df.values), index=log_results_df.index, columns=log_results_df.columns)
    results_df.plot.line()
    results_df.plot.scatter(x='Actual', y='Forecast')
    plt.show()

    print(f"[Original Space] R Squared: {r2_score(results_df['Actual'], results_df['Forecast']):.4f}, "
          f"Mean Absolute Error: {mean_absolute_error(results_df['Actual'], results_df['Forecast']):.4f}")
