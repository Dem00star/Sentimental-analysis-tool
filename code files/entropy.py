from scipy.stats import entropy
import numpy as np

def calculate_entropy(probabilities):
    """
    Calculates entropy of the probability distribution.
    """
    # Ensure probabilities sum to 1
    if np.sum(probabilities) == 0:
        raise ValueError("Probabilities must sum to a non-zero value.")

    # Return scalar entropy
    return float(entropy(probabilities))  # Force return type to float