import numpy as np
import skfuzzy as fuzz

def apply_fuzzy_logic(sentiment_scores, weight_strategy='balanced'):
    """
    More robust fuzzy logic rating calculation
    
    Args:
        sentiment_scores (list): [positive, neutral, negative] sentiment probabilities
        weight_strategy (str): Weighting approach
    
    Returns:
        float: Rating between 1-5
    """
    # Validate input
    if not isinstance(sentiment_scores, (list, np.ndarray)) or len(sentiment_scores) != 3:
        raise ValueError("Sentiment scores must be a list of three probabilities")
    
    # Ensure probabilities sum close to 1
    if not np.isclose(sum(sentiment_scores), 1.0, atol=0.1):
        # Normalize if not already normalized
        sentiment_scores = np.array(sentiment_scores) / np.sum(sentiment_scores)
    
    # Weight strategies with more nuanced approach
    strategies = {
        'conservative': [0.3, 1.0, 5.0],   # More cautious
        'balanced': [0.5, 0.8, 4.0],       # Moderate approach
        'optimistic': [0.7, 0.6, 3.0]      # More generous
    }
    
    # Select weights
    weights = strategies.get(weight_strategy, strategies['balanced'])
    
    # Calculate base rating
    base_rating = (
        sentiment_scores[0] * weights[0] +  # Positive impact
        sentiment_scores[1] * weights[1] +  # Neutral dampening
        (1 - sentiment_scores[2]) * weights[2]  # Negative inverse
    )
    
    # Non-linear scaling to 1-5 range
    final_rating = np.clip(base_rating * 4 + 1, 1, 5)
    
    return round(final_rating, 1)

# Test function
if __name__ == "__main__":
    # Test various sentiment scenarios
    test_cases = [
        ([0.8, 0.15, 0.05], 'optimistic'),   # Very positive
        ([0.3, 0.4, 0.3], 'balanced'),       # Neutral
        ([0.1, 0.2, 0.7], 'conservative'),   # Negative
    ]
    
    for scores, strategy in test_cases:
        print(f"Scores: {scores}, Strategy: {strategy}")
        print(f"Rating: {apply_fuzzy_logic(scores, strategy)}\n")
