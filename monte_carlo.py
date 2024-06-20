import os
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from typing import List, Optional
from progress_bar import ProgressBar  # Assuming progress_bar is available


class MonteCarloSimulation:
    def __init__(
        self,
        stocks: List[str],
        end_date: dt.datetime = dt.datetime.now(),
        start_date_offset: int = 300,
    ) -> None:
        self.stocks = stocks
        self.start_date = end_date - dt.timedelta(days=start_date_offset)
        self.mean_returns, self.cov_matrix, self.returns = self.load_data(
            self.stocks, self.start_date, end_date
        )

        self.default_weight = np.random.random(len(self.mean_returns))
        self.default_weight /= np.sum(self.default_weight)

    @staticmethod
    def load_data(stocks: List[str], start: dt.datetime, end: dt.datetime):
        try:
            stock_data = yf.download(stocks, start=start, end=end)["Close"]
            returns = stock_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            return mean_returns, cov_matrix, returns
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.Series(), pd.DataFrame(), pd.DataFrame()

    def simulate(
        self,
        num_simulations: int = 1000,
        num_days: int = 252,
        weights: Optional[List[float]] = None,
    ) -> pd.DataFrame:
        if weights is None:
            weights = self.default_weight

        mean_matrix = np.full(
            shape=(num_days, len(weights)), fill_value=self.mean_returns
        )

        simulated_price_paths = np.zeros((num_days, num_simulations))

        # Initialize the progress bar
        p_bar = ProgressBar(num_simulations, os.path.basename(__file__), bar_length=100)

        L = np.linalg.cholesky(self.cov_matrix)
        for i in range(num_simulations):
            Z = np.random.normal(size=(num_days, len(weights)))
            daily_returns = mean_matrix + np.dot(Z, L.T)
            simulated_price_paths[:, i] = np.cumprod(np.dot(daily_returns, weights) + 1)
            p_bar.increment()

        return pd.DataFrame(simulated_price_paths)

    @staticmethod
    def value_at_risk(returns: np.ndarray, percentile: int = 5) -> float:
        return np.percentile(returns, percentile)

    def cond_value_at_risk(self, returns: np.ndarray, percentile: int = 5) -> float:
        var_threshold = self.value_at_risk(returns, percentile)
        return returns[returns <= var_threshold].mean()
