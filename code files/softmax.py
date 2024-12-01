import numpy as np

def softmax(logits):
    """
    Converts logits into probabilities using the Softmax function.
    """
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    return exp_logits / np.sum(exp_logits)