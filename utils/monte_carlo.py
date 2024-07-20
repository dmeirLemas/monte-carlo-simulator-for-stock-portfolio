import datetime as dt
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class MonteCarloSimulation:
    def __init__(
        self,
        stocks: List[str],
        end_date: dt.datetime = dt.datetime.now(),
        start_date_offset: int = 300,
    ) -> None:
        self.stocks = stocks
        self.start_date = end_date - dt.timedelta(days=start_date_offset)
        self.mean_returns, self.cov_matrix, self.returns, self.var_returns = (
            self.load_data(self.stocks, self.start_date, end_date)
        )

        self.default_weight = np.random.random(len(self.mean_returns))
        self.default_weight /= np.sum(self.default_weight)

        print(f"Starting Date: {self.start_date}, End Date: {end_date}")

    @staticmethod
    def load_data(stocks: List[str], start: dt.datetime, end: dt.datetime):
        try:
            stock_data = yf.download(stocks, start=start, end=end)["Close"]
            returns = stock_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            var_return = returns.var()
            return mean_returns, cov_matrix, returns, var_return
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.Series(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def simulate(
        self,
        num_simulations: int = 1000,
        num_days: int = 252,
        weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        if weights is None:
            weights = self.default_weight

        mean_matrix = np.full(
            shape=(num_days, len(weights)), fill_value=self.mean_returns
        )

        mu = np.dot(self.mean_returns, weights)
        b = np.dot(np.sqrt(self.var_returns / 2), weights)

        simulated_price_paths = np.zeros((num_days, num_simulations))

        L = np.linalg.cholesky(self.cov_matrix)
        for i in range(num_simulations):
            Z = np.random.laplace(mu, b, size=(num_days, len(weights)))
            daily_returns = mean_matrix + np.dot(Z, L.T)
            simulated_price_paths[:, i] = np.cumprod(np.dot(daily_returns, weights) + 1)

        return pd.DataFrame(simulated_price_paths)
