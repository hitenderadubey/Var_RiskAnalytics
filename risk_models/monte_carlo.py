import numpy as np
from scipy.stats import norm

def monte_carlo_var(
    mean_return: float, 
    volatility: float, 
    horizon: int = 1,
    simulations: int = 10_000,
    confidence: float = 0.95
) -> float:
    """
    Simulates future portfolio returns using Geometric Brownian Motion.
    Args:
        mean_return: Expected daily return (e.g., 0.001 for 0.1%).
        volatility: Daily volatility (e.g., 0.02 for 2%).
        horizon: Days to project.
        simulations: Number of MC paths.
    Returns:
        VaR estimate.
    """
    # Generate random paths
    daily_returns = norm.rvs(
        loc=mean_return, 
        scale=volatility, 
        size=(horizon, simulations)
    )
    
    # Cumulative returns
    cumulative_returns = np.prod(1 + daily_returns, axis=0) - 1
    
    # VaR calculation
    var = -np.percentile(cumulative_returns, 100 * (1 - confidence))
    return var