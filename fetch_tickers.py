import yfinance as yf
import pandas as pd
import os
from progress_bar import ProgressBar


def get_sp500_tickers():
    # Fetch the S&P 500 tickers from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].tolist()
    return tickers


def get_company_info(tickers):
    company_info = []
    p_bar = ProgressBar(len(tickers), os.path.basename(__file__), bar_length=100)
    for ticker in tickers:
        try:
            ticker_data = yf.Ticker(ticker)
            short_name = ticker_data.info.get("shortName", "N/A")
            company_info.append((ticker, short_name))
            p_bar.increment()
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            with open("log.txt", "w") as f:
                f.writelines(f"Error fetching data for {ticker}: {e}\n")
    return company_info


if __name__ == "__main__":
    sp500_tickers = get_sp500_tickers()
    sp500_company_info = get_company_info(sp500_tickers)

    # Print the first 10 tickers and their short names
    with open("sp500_company_info.txt", "w") as f:
        for line in sp500_company_info:
            f.writelines(str(line) + "\n")
