from scipy.stats import entropy
import numpy as np

def calculate_entropy(probabilities):
    """
    Calculates entropy of the probability distribution.
    """
    # Ensure probabilities sum to 1
    if np.sum(probabilities) == 0:
        raise ValueError("Probabilities must sum to a non-zero value.")
    
    return entropy(probabilities)

def compute_dynamic_threshold(entropy_values):
    """
    Computes a dynamic threshold based on the mean and standard deviation of entropy values.
    """
    if len(entropy_values) == 0:
        raise ValueError("Entropy values list cannot be empty.")
        
    mean_entropy = np.mean(entropy_values)
    std_entropy = np.std(entropy_values)
    
    threshold = mean_entropy + 0.5 * std_entropy  # Adjust multiplier as needed
    return threshold