import os
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from progress_bar import ProgressBar
from typing import List


class MonteCarloSimulation:
    def __init__(
        self,
        stocks: List[str],
        endDate: dt.datetime = dt.datetime.now(),
        startDateOffset: int = 300,
    ) -> None:
        # self.stocks = [stock + ".AX" for stock in stocks]
        self.stocks = stocks
        self.startDate = endDate - dt.timedelta(days=startDateOffset)
        self.endDate = endDate
        self.meanReturns, self.covMatrix = self.load_data(
            self.stocks, self.startDate, self.endDate
        )

        self.default_weight = np.random.random(len(self.meanReturns))
        self.default_weight /= np.sum(self.default_weight)

    @staticmethod
    def load_data(stocks: List[str], start: dt.datetime, end: dt.datetime):
        stockData = yf.download(stocks, start=start, end=end)["Close"]
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()

        return meanReturns, covMatrix

    def simulate(
        self,
        num_simulations: int = 1000,
        num_days: int = 252,
        weights: List[float] = None,
    ) -> pd.DataFrame:
        if weights is None:
            weights = self.default_weight

        meanMatrix = np.full(
            shape=(num_days, len(weights)), fill_value=self.meanReturns
        ).T
        meanMatrix = meanMatrix.T

        simulated_price_paths = np.zeros((num_days, num_simulations))

        p_bar = ProgressBar(num_simulations, os.path.basename(__file__), bar_length=100)

        for i in range(num_simulations):
            Z = np.random.normal(size=(num_days, len(weights)))
            L = np.linalg.cholesky(self.covMatrix)
            daily_returns = meanMatrix + np.dot(Z, L.T)
            simulated_price_paths[:, i] = np.cumprod(np.dot(daily_returns, weights) + 1)
            p_bar.increment()

        return pd.DataFrame(simulated_price_paths)
