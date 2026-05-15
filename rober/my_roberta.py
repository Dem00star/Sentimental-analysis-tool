import pandas as pd
import logging
from transformers import pipeline
from preprocessing import preprocess_data, group_by_course

# Optimized Sentiment analysis with IMDb fine-tuned model
class IMDBSentimentScorer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Initialize the IMDbSentimentScorer with a pretrained DistilBERT model fine-tuned on IMDb sentiment tasks.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        try:
            self.classifier = pipeline('sentiment-analysis', model=model_name, device=0)
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
            raise

    def calculate_sentiment_for_tokens(self, tokens):
        """
        Calculate sentiment scores for a batch of tokens at once.

        Args:
            tokens (list): A list of words (tokens) to analyze.

        Returns:
            list: A list of sentiment probabilities for each word.
        """
        if not tokens:
            return []  # Handle empty token lists gracefully

        # Batch inference to reduce the number of API calls
        sentiments = self.classifier(tokens, truncation=True)
        
        # Extracting the probability scores (confidence values)
        sentiment_scores = []
        for sentiment in sentiments:
            # POSITIVE has a score of 1, NEGATIVE has a score of -1
            score = sentiment['score'] if sentiment['label'] == 'POSITIVE' else -sentiment['score']
            sentiment_scores.append(score)

        return sentiment_scores

    def _preprocess_review(self, review, max_length=512):
        """
        Preprocess a single review with safety checks.
        """
        review = str(review).strip()[:max_length]
        
        if not review:
            return None
        
        # Tokenize using the classifier's tokenizer
        tokens = self.classifier.tokenizer(
            review, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        return review, tokens

    def process_reviews(self, reviews, max_length=512):
        """
        Process multiple reviews with comprehensive error handling
        """
        filtered_reviews = []
        sentiment_scores = []
        
        for i, review in enumerate(reviews, 1):
            try:
                preprocessed = self._preprocess_review(review, max_length)
                
                if not preprocessed:
                    self.logger.warning(f"Empty review skipped at index {i}")
                    continue
                
                original_review, tokens = preprocessed
                scores = self.calculate_sentiment_for_tokens(tokens['input_ids'][0])  # Extract the actual tokens
                
                filtered_reviews.append(original_review)
                sentiment_scores.append(scores)
                
            except Exception as e:
                self.logger.error(f"Error processing review {i}: {e}")
        
        self.logger.info(f"Total Reviews Processed: {len(filtered_reviews)}")
        
        return filtered_reviews, sentiment_scores

# Function to process sentiment for a CSV file
def process_sentiment(file_path, batch_size=1000):
    """
    Process sentiment scores for a given CSV file in batches to handle large datasets efficiently.

    Args:
        file_path (str): Path to the CSV file containing the data.
        batch_size (int): Number of rows to process in each batch.

    Returns:
        pd.DataFrame: DataFrame containing the sentiment scores for each course.
    """
def roberta_score():
    # Read the data from the CSV file
    data = pd.read_csv('test-.csv')
    
    # Preprocess the data (e.g., tokenization, cleaning)
    preprocessed_data = preprocess_data(data)

    # Step 2: Initialize IMDb sentiment scorer
    scorer = IMDBSentimentScorer()

    # Group tokens by CourseId
    grouped_data = group_by_course(preprocessed_data)

    # Step 3: Calculate sentiment for each course (without batching)
    grouped_data['sentiment_scores'] = grouped_data['tokens'].apply(
        lambda tokens: scorer.calculate_sentiment_for_tokens(tokens)
    )

    # Filter out rows with empty sentiment scores
    filtered_batch = grouped_data[grouped_data['sentiment_scores'].apply(lambda x: len(x) > 0)]

    print(f"Processed {len(filtered_batch)} rows successfully.")

    # Return the final result
    return filtered_batch

c= roberta_score()
print ("                       senti score                                 ")
print (c)