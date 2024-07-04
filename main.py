from monte_carlo import MonteCarloSimulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from progress_bar import ProgressBar
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_simulation(simulation, num_simulations, num_days, weights):
    simulation_results = simulation.simulate(
        num_simulations=num_simulations, num_days=num_days, weights=weights
    )
    final_prices = simulation_results.iloc[-1, :]
    return final_prices


def simulate_monte_carlo(num_simulations, stocks, num_days):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)

    final_prices_list = []

    p_bar = ProgressBar(total=num_simulations * 1000, program_name="SIMULATOR")

    simulation = MonteCarloSimulation(stocks)

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
            p_bar.increment()

    final_prices_df = pd.concat([pd.Series(fp) for fp in final_prices_list], axis=1)

    return final_prices_df


def main():
    num_simulations = 100
    stocks = ["GOOGL", "AMZN", "MSFT"]
    num_days = 260

    final_prices_df = simulate_monte_carlo(
        num_simulations=num_simulations, stocks=stocks, num_days=num_days
    )

    CLT_final_prices = final_prices_df.mean()

    print(CLT_final_prices)

    print("Expected Return:", CLT_final_prices.mean())
    print("Standard Deviation", CLT_final_prices.std())

    plt.hist(CLT_final_prices, bins=50, edgecolor="black")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()
