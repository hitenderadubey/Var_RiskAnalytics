import numpy as np
import pandas as pd
from typing import Dict, Tuple

def calculate_historical_var(
    prices: pd.DataFrame, 
    portfolio: Dict[str, float], 
    confidence: float = 0.95
) -> Tuple[float, pd.DataFrame]:
    """
    Args:
        prices: DataFrame with dates as index and assets as columns.
        portfolio: Dict of {asset: weight} (e.g., {"AAPL": 0.6, "MSFT": 0.4}).
        confidence: Confidence level (e.g., 0.95 for 95% VaR).
    """
    # 1. Calculate returns and drop NaN
    returns = prices.pct_change().dropna()
    
    # 2. Align portfolio weights with returns columns
    weights = pd.Series(portfolio)
    weighted_returns = returns * weights
    
    # 3. Sum weighted returns
    portfolio_returns = weighted_returns.sum(axis=1)
    
    # 4. Calculate VaR
    var = -np.percentile(portfolio_returns, 100 * (1 - confidence))
    return var, portfolio_returns