# preprocessing.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Make sure you have these NLTK data downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

def preprocess_review_text(review):
    """
    Clean and normalize review text
    """
    import re
    # Convert to lowercase and remove special characters
    review = re.sub(r'[^a-zA-Z\s]', '', str(review).lower())
    return review

def preprocess_data(data):
    """
    Enhanced data preprocessing with robust error handling
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def safe_process(review):
        try:
            # Process text
            processed = ' '.join(
                lemmatizer.lemmatize(word.lower())
                for word in word_tokenize(preprocess_review_text(review))
                if word.lower() not in stop_words
            )
            return processed
        except Exception as e:
            print(f"Processing error: {e}")
            return ""

    # Apply safe processing to the DataFrame
    data['processed'] = data['Review'].apply(safe_process)
    data['tokens'] = data['processed'].apply(word_tokenize)
    return data

def group_by_course(data):
    """
    Groups the data by CourseId and aggregates the tokens.
    """
    # Group by 'CourseId' and aggregate the tokens
    grouped_data = data.groupby('CourseId')['tokens'].apply(lambda x: [item for sublist in x for item in sublist]).reset_index()
    return grouped_data
