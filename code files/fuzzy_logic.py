import numpy as np
import skfuzzy as fuzz

def apply_fuzzy_logic(sentiment_scores):
    """
    Applies fuzzy logic to determine the rating (1 to 5) for the course based on sentiment scores.
    
    Parameters:
        sentiment_scores (list or np.ndarray): List or array containing three elements: 
                                                [positive, neutral, negative].
    
    Returns:
        float: Rounded rating from 1 to 5.
    """
    
    # Check if sentiment_scores is valid
    if not isinstance(sentiment_scores, (list, np.ndarray)) or len(sentiment_scores) != 3:
        raise ValueError("Sentiment scores must be a list or array of three elements: [positive, neutral, negative].")

    # Define fuzzy sets for sentiment scores (0 to 1 scale) and ratings (1 to 5)
    sentiment_range = np.arange(0, 1.1, 0.1)  # Sentiment score range (0 to 1 scale)
    rating_range = np.arange(1, 6, 1)  # Rating range (1 to 5)

    # Define fuzzy membership functions for sentiment categories (positive, neutral, negative)
    positive = fuzz.trimf(sentiment_range, [0.6, 1, 1])  # Positive sentiment
    neutral = fuzz.trimf(sentiment_range, [0.3, 0.5, 0.7])  # Neutral sentiment
    negative = fuzz.trimf(sentiment_range, [0, 0, 0.4])  # Negative sentiment

    # Calculate fuzzy membership for each sentiment score
    positive_membership = fuzz.interp_membership(sentiment_range, positive, sentiment_scores[0])
    neutral_membership = fuzz.interp_membership(sentiment_range, neutral, sentiment_scores[1])
    negative_membership = fuzz.interp_membership(sentiment_range, negative, sentiment_scores[2])

    # Debugging output
    print(f"Positive Membership: {positive_membership}")
    print(f"Neutral Membership: {neutral_membership}")
    print(f"Negative Membership: {negative_membership}")

    # Create an array for combined membership corresponding to rating range
    combined_array = np.array([
        negative_membership * 1,
        neutral_membership * 3,
        neutral_membership * 3 + positive_membership * 5,
        positive_membership * 5,
        positive_membership * 5
    ])

    print(f"Combined Membership Array: {combined_array}")

    # Check if combined_array has valid values before defuzzification
    if np.all(combined_array <= 0):
        raise ValueError("Combined membership array must contain valid values for defuzzification.")

    # Apply defuzzification using the centroid method to determine the final rating
    final_rating = fuzz.defuzz(rating_range, combined_array, 'centroid')

    return round(final_rating, 1)  # Round to one decimal place for rating