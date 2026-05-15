# my_senticnet.py
import pandas as pd
from preprocessing import read_data, preprocess_data, group_by_course
from nltk.stem import WordNetLemmatizer
import numpy as np

# Import the senticnet dictionary (skip errors)
senticnet = {}
try:
    # Attempt to import the senticnet dictionary from the external file
    import senticnet_file

    # Manually load the senticnet dictionary if it's not properly loaded
    senticnet.update(senticnet_file.senticnet)
except Exception as e:
    print(f"Error loading senticnet: {e}")

# SenticNet Sentiment Analysis Class
class SenticnetScorer:
    def __init__(self):
        """
        Initialize the SenticnetScorer with a WordNet lemmatizer.
        """
        self.lemmatizer = WordNetLemmatizer()

    def get_word_sentiment(self, word):
        """
        Get the polarity score of a word using SenticNet.

        Args:
            word (str): The word for which sentiment is to be calculated.

        Returns:
            float: The polarity score for the word.
        """
        lemma = self.lemmatizer.lemmatize(word.lower())

        # Check if the word is in the senticnet dictionary
        try:
            if lemma in senticnet:
                return senticnet[lemma][7]  # polarity_value is at index 7
            else:
                return None  # Return None if the word is not found
        except KeyError as e:
            # If there's a KeyError (invalid key), we simply skip this word
            print(f"Skipping word '{word}' due to KeyError: {e}")
            return None

    def calculate_sentiment_for_tokens(self, tokens):
        """
        Calculate sentiment scores for a list of tokens (words).

        Args:
            tokens (list): A list of words (tokens) to analyze.

        Returns:
            list: A list of sentiment scores for each word.
        """
        sentiments = []
        for token in tokens:
            sentiment = self.get_word_sentiment(token)
            if sentiment is not None:  # Only add sentiment if it's not None
                sentiments.append(sentiment)
        return sentiments


# Function to process sentiment for a CSV file
def process_sentiment(file_path):
    """
    Process sentiment scores for a given CSV file, calculate sentiment for each word and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: DataFrame containing the sentiment scores for each course.
    """
    # Step 1: Load and preprocess the data
    data = read_data(file_path)
    preprocessed_data = preprocess_data(data)

    # Step 2: Group tokens by CourseId
    grouped_data = group_by_course(preprocessed_data)

    # Step 3: Initialize SenticNet scorer
    scorer = SenticnetScorer()

    # Step 4: Calculate sentiment scores for each course
    def calculate_sentiment_for_course(tokens):
        sentiment_scores = scorer.calculate_sentiment_for_tokens(tokens)
        return sentiment_scores

    grouped_data['sentiment_scores'] = grouped_data['tokens'].apply(calculate_sentiment_for_course)

    # Step 5: Filter out rows with empty sentiment scores (those with no valid polarity)
    filtered_data = grouped_data[grouped_data['sentiment_scores'].apply(lambda x: len(x) > 0)]

    return filtered_data

v = process_sentiment('test-.csv')
print(v)

