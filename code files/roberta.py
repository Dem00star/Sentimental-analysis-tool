import torch
import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import entropy

class SentimentAnalyzer:
    def __init__(self, model_name="cardiffnlp/twitter-roberta-base-sentiment"):
        """
        Initialize RoBERTa sentiment analyzer with comprehensive setup
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
            raise

    def _preprocess_review(self, review, max_length=512):
        """
        Preprocess a single review with safety checks
        """
        review = str(review).strip()[:max_length]
        
        if not review:
            return None
        
        tokens = self.tokenizer(
            review, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        )
        
        return review, tokens

    def _calculate_sentiment(self, tokens):
        """
        Calculate sentiment probabilities and entropy
        """
        with torch.no_grad():
            outputs = self.model(**tokens)
            logits = outputs.logits.squeeze()
        
        # Softmax with numerical stability
        exp_logits = np.exp(logits.numpy() - np.max(logits.numpy()))
        probabilities = exp_logits / np.sum(exp_logits)
        
        # Calculate entropy
        word_entropy = float(entropy(probabilities))
        
        return probabilities.tolist(), word_entropy

    def process_reviews(self, reviews, max_length=512, entropy_threshold=1.5):
        """
        Process multiple reviews with comprehensive error handling
        """
        filtered_reviews = []
        entropy_values = []
        sentiment_scores = []
        
        for i, review in enumerate(reviews, 1):
            try:
                preprocessed = self._preprocess_review(review, max_length)
                
                if not preprocessed:
                    self.logger.warning(f"Empty review skipped at index {i}")
                    continue
                
                original_review, tokens = preprocessed
                probabilities, word_entropy = self._calculate_sentiment(tokens)
                
                # Additional filtering based on entropy
                if word_entropy < entropy_threshold:
                    filtered_reviews.append(original_review)
                    entropy_values.append(word_entropy)
                    sentiment_scores.append(probabilities)
                
            except Exception as e:
                self.logger.error(f"Error processing review {i}: {e}")
        
        self.logger.info(f"Total Reviews Processed: {len(filtered_reviews)}")
        
        return filtered_reviews, entropy_values, sentiment_scores

# Demonstration script
if __name__ == "__main__":
    import pandas as pd
    
    try:
        df = pd.read_csv('test.csv')
        analyzer = SentimentAnalyzer()
        
        course_groups = df.groupby('CourseId')
        
        for course_id, group in course_groups:
            print(f"\nProcessing CourseId: {course_id}")
            reviews = group['Review'].tolist()
            
            filtered_reviews, entropy_values, sentiment_scores = analyzer.process_reviews(reviews)
            
            print(f"Filtered Reviews: {len(filtered_reviews)}")
            for review, score in zip(filtered_reviews, sentiment_scores):
                print(f"Review: {review}")
                print(f"Sentiment: {score}\n")
    
    except Exception as e:
        print(f"Script execution error: {e}")
