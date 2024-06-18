# Example Usage Of The Simulation
from monte_carlo import MonteCarloSimulation
import matplotlib.pyplot as plt

simulation = MonteCarloSimulation(["AAPL", "MSFT"])
simulation_results = simulation.simulate(num_simulations=1000, num_days=260)

final_prices = simulation_results.iloc[-1, :]

mean_final_price = final_prices.mean()
std_final_price = final_prices.std()

value_at_risk_5_percent = simulation.value_at_risk(simulation_results.iloc[-1])
cond_value_at_risk_5_percent = simulation.cond_value_at_risk(
    simulation_results.iloc[-1]
)

plt.plot(simulation_results)
plt.show()
plt.clf()

print(f"Mean of Final Prices: {mean_final_price}")
print(f"Standard Deviation of Final Prices: {std_final_price}")

print("value_at_risk_5_percent: ", value_at_risk_5_percent)
print("cond_value_at_risk_5_percent: ", cond_value_at_risk_5_percent)

plt.hist(final_prices, bins=50, edgecolor="black")
plt.xlabel("Final Prices")
plt.ylabel("Frequency")
plt.title("Histogram of Final Prices from Monte Carlo Simulations")
plt.show()
