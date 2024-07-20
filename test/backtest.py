import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
from utils.monte_carlo import MonteCarloSimulation
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_simulation(simulation, num_simulations, num_days, weights):
    simulation_results = simulation.simulate(
        num_simulations=num_simulations, num_days=num_days, weights=weights
    )
    final_prices = simulation_results.iloc[-1, :]
    return final_prices


def simulate_monte_carlo(
    num_simulations, stocks, num_days, weights, end_date, start_date_offset
):
    final_prices_list = []

    simulation = MonteCarloSimulation(
        stocks=stocks, end_date=end_date, start_date_offset=start_date_offset
    )

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_simulation, simulation, num_simulations, num_days, weights
            )
            for _ in range(num_simulations * 1000)
        ]

        for future in as_completed(futures):
            final_prices = future.result()
            final_prices_list.append(final_prices)

    final_prices_df = pd.concat([pd.Series(fp) for fp in final_prices_list], axis=1)

    return final_prices_df, weights


def get_historical_data(stocks, start_date, end_date):
    stock_data = yf.download(stocks, start=start_date, end=end_date)["Close"]
    return stock_data


def run_backtest(
    stocks,
    overall_start_date,
    overall_end_date,
    training_duration,
    testing_duration,
    num_simulations,
    num_days,
):
    all_results = []

    current_start_date = overall_start_date
    while current_start_date + training_duration + testing_duration <= overall_end_date:
        training_end_date = current_start_date + training_duration
        testing_end_date = training_end_date + testing_duration

        print(
            f"Training from {current_start_date} to {training_end_date}, testing from {training_end_date} to {testing_end_date}"
        )

        training_data = get_historical_data(
            stocks, current_start_date, training_end_date
        )
        testing_data = get_historical_data(stocks, training_end_date, testing_end_date)

        if training_data.empty or testing_data.empty:
            current_start_date += training_duration
            continue

        weights = np.random.random(len(stocks))
        weights /= np.sum(weights)

        print("*" * 50)
        print(weights)
        print("*" * 50)

        final_prices_df, weights = simulate_monte_carlo(
            num_simulations=num_simulations,
            stocks=stocks,
            num_days=num_days,
            weights=weights,
            end_date=training_end_date,
            start_date_offset=training_duration.days,
        )

        CLT_final_prices = final_prices_df.mean()

        initial_prices = testing_data.iloc[0]
        final_prices = testing_data.iloc[-1]
        actual_returns = (final_prices / initial_prices) - 1
        actual_portfolio_return = np.dot(actual_returns, weights) + 1

        all_results.append((CLT_final_prices.mean(), actual_portfolio_return))

        print(
            f"Simulated mean: {CLT_final_prices.mean()}, Actual mean: {actual_portfolio_return}"
        )

        current_start_date += training_duration

    return all_results


def main():
    stocks = ["GOOGL", "ABT", "MSFT"]
    overall_start_date = dt.datetime.now() - dt.timedelta(days=3 * 365)
    overall_end_date = dt.datetime.now()
    training_duration = dt.timedelta(days=180)
    testing_duration = dt.timedelta(days=180)
    num_simulations = 1000
    num_days = 180

    results = run_backtest(
        stocks,
        overall_start_date,
        overall_end_date,
        training_duration,
        testing_duration,
        num_simulations,
        num_days,
    )

    simulated_means, actual_means = zip(*results)

    plt.plot(simulated_means, label="Simulated Mean")
    plt.plot(actual_means, label="Actual Mean")
    plt.xlabel("Backtest Periods")
    plt.ylabel("Mean Prices")
    plt.legend()
    plt.show()

    plt.hist(
        np.array(simulated_means) - np.array(actual_means), bins=50, edgecolor="black"
    )
    plt.xlabel("Difference between Simulated and Actual Means")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
