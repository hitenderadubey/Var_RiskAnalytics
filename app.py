import pandas as pd
from risk_models.var import calculate_historical_var
from risk_models.monte_carlo import monte_carlo_var
from risk_models.stress_test import stress_test_crash
import matplotlib.pyplot as plt

def main():
    # Load data
    prices = pd.read_csv("data/stock_prices.csv", index_col="date", parse_dates=True)
    portfolio = {"AAPL": 0.6, "MSFT": 0.4}
    
    # 1. Historical VaR
    var, returns = calculate_historical_var(prices, portfolio)
    print(f"Historical 95% VaR: {var:.2%}")
    
    # 2. Monte Carlo VaR
    mean_return = returns.mean()
    volatility = returns.std()
    mc_var = monte_carlo_var(mean_return, volatility)
    print(f"Monte Carlo 95% VaR: {mc_var:.2%}")
    
    # 3. Stress Test
    crash_loss = stress_test_crash(returns.values)
    print(f"Stress Test (20% Crash) Loss: {crash_loss:.2%}")
    
    # 4. Plot returns distribution
    plt.figure(figsize=(10, 6))
    plt.hist(returns, bins=50, alpha=0.7, label="Daily Returns")
    plt.axvline(-var, color="red", linestyle="--", label="95% VaR")
    plt.title("Portfolio Returns Distribution")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()