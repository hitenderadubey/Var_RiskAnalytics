import numpy as np

def stress_test_crash(
    returns: np.ndarray, 
    crash_scenario: float = -0.20  # 20% market drop
) -> float:
    """
    Stress test: What if the market crashes?
    Args:
        returns: Historical returns.
        crash_scenario: Simulated loss (e.g., -0.20 for -20%).
    Returns:
        Worst-case portfolio loss.
    """
    stressed_returns = returns.copy()
    stressed_returns[-1] = crash_scenario  # Apply crash to the last day
    return np.sum(stressed_returns)