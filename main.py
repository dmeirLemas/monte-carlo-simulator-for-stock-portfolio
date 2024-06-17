# Example Usage Of The Simulation


from monte_carlo import MonteCarloSimulation
import matplotlib.pyplot as plt

simulation = MonteCarloSimulation(["GOOG", "AMZN", "ADBE"])
simulation_results = simulation.simulate(num_simulations=1000000, num_days=260)

final_prices = simulation_results.iloc[-1, :]

mean_final_price = final_prices.mean()
std_final_price = final_prices.std()

print(f"Mean of Final Prices: {mean_final_price}")
print(f"Standard Deviation of Final Prices: {std_final_price}")

plt.hist(final_prices, bins=50, edgecolor="black")
plt.xlabel("Final Prices")
plt.ylabel("Frequency")
plt.title("Histogram of Final Prices from Monte Carlo Simulations")
plt.show()
