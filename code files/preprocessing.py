import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def translate_to_english(text):
    """
    Translates text to English with length and error handling.
    """
    try:
        # Truncate text if too long before translation
        text = text[:5000]
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text[:5000]  # Fallback and truncate

def preprocess_data(data):
    """
    Enhanced data preprocessing with robust error handling
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def safe_process(review):
        try:
            # Ensure review is a string and not too long
            review = str(review)[:5000]
            
            # Translate
            translated = translate_to_english(review)
            
            # Process
            processed = ' '.join(
                lemmatizer.lemmatize(word.lower())
                for word in word_tokenize(preprocess_review_text(translated))
                if word.lower() not in stop_words
            )
            return processed
        except Exception as e:
            print(f"Processing error: {e}")
            return ""

    # Apply safe processing
    data['processed'] = data['Review'].apply(safe_process)
    data['tokens'] = data['processed'].apply(word_tokenize)
    return data
