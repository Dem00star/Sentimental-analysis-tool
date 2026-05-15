import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_entropy_for_score(score):
    """
    Calculate the Shannon entropy for a single sentiment score.
    
    Args:
        score (float): A sentiment score (e.g., 0.5, -0.1, etc.).
        
    Returns:
        float: Entropy value for the sentiment score.
    """
    try:
        # Normalize the score in the range of [-1, 1]
        if score == 0:
            return 1.0  # Return maximum entropy for neutral sentiment (score == 0)

        # Map sentiment score to a probability range [0, 1]
        prob = (score + 1) / 2  
        
        # Avoid division by zero or log2(0) issues
        if prob == 0 or prob == 1:
            return 0.0  # Zero entropy when score is at extreme (either fully positive or fully negative)

        entropy_value = -prob * np.log2(prob) - (1 - prob) * np.log2(1 - prob)

        return entropy_value
    except Exception as e:
        print(f"Error calculating entropy for score {score}: {str(e)}")
        return 0.0

def add_entropy_column(input_file, output_file):
    """
    Add an entropy_value column to the CSV and save the new file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to save the output CSV file.
    """
    try:
        # Read the input CSV file
        data = pd.read_csv(input_file)

        # Convert string representations of lists to actual lists
        data['tokens'] = data['tokens'].apply(eval)  # List of tokens (words)
        data['sentiment_scores'] = data['sentiment_scores'].apply(eval)  # List of sentiment scores

        # Ensure the 'tokens' and 'sentiment_scores' lists are of equal length
        if len(data['tokens']) != len(data['sentiment_scores']):
            raise ValueError("Mismatch between number of tokens and sentiment scores.")

        # Calculate entropy for each word's sentiment score
        entropy_values = [
            [calculate_entropy_for_score(score) for score in row] 
            for row in data['sentiment_scores']
        ]
        
        # Flatten the entropy values to match the structure
        data['entropy_values'] = entropy_values

        # Save the updated DataFrame to a new CSV file
        data.to_csv(output_file, index=False)
        print(f"Updated CSV file with entropy values saved to {output_file}.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    input_csv = "processed_pl.csv"  # Input file path
    output_csv = "entropy_pl.csv"  # Output file path

    add_entropy_column(input_csv, output_csv)