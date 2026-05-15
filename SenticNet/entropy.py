import pandas as pd
import numpy as np
from my_senticnet import SenticnetScorer  # Assuming you already have SenticnetScorer class

# Define the entropy computation using the Shannon formula for sentiment scores
def compute_entropy(sentiment_scores):
    """
    Compute the entropy for a list of sentiment scores.

    Args:
        sentiment_scores (list): A list of sentiment scores.

    Returns:
        float: The entropy value for the list of sentiment scores.
    """
    sentiment_scores = [score for score in sentiment_scores if score is not None]
    
    if len(sentiment_scores) == 0:
        return 0.0
    
    values, counts = np.unique(sentiment_scores, return_counts=True)
    probabilities = counts / len(sentiment_scores)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def process_sentiment(file_path, entropy_threshold=0.7):
    """
    Read the CSV, process it using Senticnet Scorer, and compute entropy for each word's sentiment.
    
    Args:
        file_path (str): The path to the CSV file.
        entropy_threshold (float): Threshold for entropy. Words with entropy above this value will be removed.
    
    Returns:
        pd.DataFrame: A DataFrame containing CourseId, cleaned sentiment scores, and mean label values.
    """
    # Step 1: Load the data
    data = pd.read_csv(file_path)  # Read the CSV into a DataFrame
    
    # Step 2: Tokenize the reviews (assuming each review is a sentence to be tokenized)
    data['tokens'] = data['Review'].apply(lambda x: str(x).split() if isinstance(x, str) else [])  # Split review text into tokens
    
    # Step 3: Initialize SenticnetScorer to compute sentiment scores
    scorer = SenticnetScorer()

    # Step 4: Function to get sentiment score for each word in a review
    def get_sentiment_scores(tokens):
        return [scorer.get_word_sentiment(token) for token in tokens]

    # Step 5: Calculate sentiment scores for each word
    data['sentiment_scores'] = data['tokens'].apply(get_sentiment_scores)

    # Step 6: Function to filter out words based on entropy of their sentiment score
    def filter_high_entropy_words(tokens, sentiment_scores, threshold):
        filtered_tokens = []
        filtered_sentiment_scores = []
        for token, score in zip(tokens, sentiment_scores):
            if score is not None:  # Ensure the sentiment score exists
                entropy = compute_entropy([score])  # Calculate the entropy of the score
                if entropy < threshold:  # Only keep words with entropy below the threshold
                    filtered_tokens.append(token)
                    filtered_sentiment_scores.append(score)
        return filtered_sentiment_scores

    # Step 7: Apply filtering to each row (course/review)
    data['filtered_sentiment_scores'] = data.apply(
        lambda row: filter_high_entropy_words(row['tokens'], row['sentiment_scores'], entropy_threshold),
        axis=1
    )

    # Step 8: Group by 'CourseId' and aggregate sentiment scores
    def group_by_course_id(df):
        """
        Groups the DataFrame by 'CourseId' and combines the sentiment scores for each course.
    
        Args:
            df (pd.DataFrame): DataFrame containing 'CourseId' and 'filtered_sentiment_scores'.
        
        Returns:
            pd.DataFrame: DataFrame grouped by 'CourseId' with aggregated sentiment scores.
        """
        def combine_sentiment_scores(group):
            # Ensure that the group is a list of lists (i.e., each row contains a list of sentiment scores)
            all_sentiment_scores = np.concatenate(group)
            return np.unique(all_sentiment_scores)

        # Group by 'CourseId' and aggregate sentiment scores
        grouped_df = df.groupby('CourseId')['filtered_sentiment_scores'].agg(lambda x: combine_sentiment_scores(list(x))).reset_index()
        return grouped_df

    # Apply the grouping function to aggregate sentiment scores
    data_by_course = group_by_course_id(data[['CourseId', 'filtered_sentiment_scores']])

    # Step 9: Compute mean labels for each course
    mean_labels = data.groupby('CourseId')['Label'].apply(np.mean).reset_index()

    # Step 10: Merge sentiment scores and mean labels into a final DataFrame
    final_df = pd.merge(data_by_course, mean_labels, on='CourseId', how='left')

    return final_df
    

def final_sentiscore():
    # Set file path (replace with actual file path)
    file_path = 'test-.csv'  # Replace with your actual file path

    # Process sentiment and calculate entropy
    result_df = process_sentiment(file_path, entropy_threshold=0.9)

    # Handling Empty Lists (if any course has no valid sentiment scores)
    result_df = result_df[result_df['filtered_sentiment_scores'].apply(len) > 0]

    # Print the resulting DataFrame with CourseId, filtered sentiment scores, and mean label
    return result_df

v= final_sentiscore()
print (v)

