import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK data is downloaded
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)  # For POS tagging

def read_data(file_path):
    """
    Reads the CSV file and returns the data as a DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error reading file: {e}")
        return pd.DataFrame()

def preprocess_review_text(review):
    """
    Cleans and normalizes review text
    """
    import re
    # Convert to lowercase and remove special characters
    review = re.sub(r'[^a-zA-Z\s]', '', str(review).lower())
    return review

def preprocess_data(data):
    """
    Enhanced data preprocessing with robust error handling, tokenization,
    lemmatization, and stopword removal.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def safe_process(review):
        try:
            # Clean and normalize the review text
            cleaned_review = preprocess_review_text(review)

            # Tokenize and process each word
            processed = ' '.join(
                lemmatizer.lemmatize(word.lower())
                for word in word_tokenize(cleaned_review)
                if word.lower() not in stop_words  # Remove stopwords
            )
            return processed
        except Exception as e:
            print(f"Processing error: {e}")
            return ""

    def extract_tokens(processed_review):
        try:
            tokens = word_tokenize(processed_review)
            # Filter tokens by part of speech (nouns, adjectives, verbs, adverbs)
            pos_tags = nltk.pos_tag(tokens)
            filtered_tokens = [
                token for token, tag in pos_tags
                if tag.startswith('N') or tag.startswith('J') or tag.startswith('V') or tag.startswith('R')
            ]
            return filtered_tokens
        except Exception as e:
            print(f"Token extraction error: {e}")
            return []

    # Apply safe processing and token extraction to the DataFrame
    data['processed'] = data['Review'].apply(safe_process)
    data['tokens'] = data['processed'].apply(extract_tokens)
    return data

def group_by_course(data):
    """
    Groups the data by CourseId and aggregates the tokens.
    """
    try:
        # Group by 'CourseId' and aggregate the tokens
        grouped_data = data.groupby('CourseId')['tokens'].apply(
            lambda x: [item for sublist in x for item in sublist]
        ).reset_index()
        return grouped_data
    except Exception as e:
        print(f"Error grouping data by course: {e}")
        return pd.DataFrame()


