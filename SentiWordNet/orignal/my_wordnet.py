import pandas as pd
from nltk.corpus import sentiwordnet as swn
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import numpy as np
from preprocessing import read_data, preprocess_data

# SentiWordNet Sentiment Analysis Class
class SentiWordNetScorer:
    def __init__(self):
        """
        Initialize the SentiWordNetScorer with a WordNet lemmatizer.
        """
        self.lemmatizer = WordNetLemmatizer()

    def get_word_sentiment(self, word):
        """
        Get the sentiment score of a word using SentiWordNet.

        Args:
            word (str): The word for which sentiment is to be calculated.

        Returns:
            float: The sentiment score for the word in the range of -1 to 1.
        """
        lemma = self.lemmatizer.lemmatize(word.lower())
        
        # Get the synsets for the word
        synsets = list(swn.senti_synsets(lemma))

        # If no synset found, return None
        if not synsets:
            return None
        
        # Calculate the average positive and negative sentiment scores
        positive_score = np.mean([synset.pos_score() for synset in synsets])
        negative_score = np.mean([synset.neg_score() for synset in synsets])
        
        # Return the sentiment score as the difference between positive and negative
        sentiment_score = positive_score - negative_score
        return sentiment_score

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
            if sentiment is not None and sentiment != 0:  # Exclude words with zero sentiment
                sentiments.append(sentiment)
        return sentiments

# Function to process sentiment for each review
def process_sentiment(file_path):
    """
    Process sentiment scores for a given CSV file and generate a DataFrame with individual reviews.

    Args:
        file_path (str): Path to the CSV file containing the data.

    Returns:
        pd.DataFrame: DataFrame containing the processed columns for each review.
    """
    # Step 1: Load and preprocess the data
    data = read_data(file_path)
    preprocessed_data = preprocess_data(data)

    # Step 2: Filter tokens by POS tagging
    def filter_tokens_by_pos(tokens):
        try:
            pos_tags = pos_tag(tokens)
            filtered_tokens = [token for token, tag in pos_tags if tag.startswith(('N', 'J', 'V', 'R'))]
            return filtered_tokens
        except Exception as e:
            print(f"Error in POS tagging: {e}")
            return []

    preprocessed_data['tokens'] = preprocessed_data['tokens'].apply(filter_tokens_by_pos)

    # Step 3: Initialize SentiWordNet scorer
    scorer = SentiWordNetScorer()

    # Step 4: Calculate sentiment scores for each review
    def calculate_sentiment(tokens):
        return scorer.calculate_sentiment_for_tokens(tokens)

    preprocessed_data['sentiment_scores'] = preprocessed_data['tokens'].apply(calculate_sentiment)

    # Step 5: Create the final DataFrame with required columns
    final_data = preprocessed_data[['CourseId', 'Review', 'tokens', 'sentiment_scores', 'Label']]
    final_data.columns = ['course', 'reviews', 'tokens', 'sentiment_scores', 'label']

    return final_data

if __name__ == "__main__":
    # Example usage
    result = process_sentiment('a.csv')
    print(result.head())
    result.to_csv('processed_pl.csv', index=False)
