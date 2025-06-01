import numpy as np
import pandas as pd
from risk_models.var import calculate_historical_var

def test_historical_var():
    # Mock data: 5 days of returns
    prices = pd.DataFrame({
        "AAPL": [100, 101, 102, 101, 100],
        "MSFT": [200, 202, 201, 205, 204]
    })
    portfolio = {"AAPL": 0.5, "MSFT": 0.5}
    var, _ = calculate_historical_var(prices, portfolio)
    assert var > 0  # VaR should be positive (loss)