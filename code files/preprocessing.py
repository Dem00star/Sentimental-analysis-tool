import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Function for preprocessing the data
def preprocess_data(data):
    # Initialize the lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Preprocessing: Remove stopwords, lemmatize, and convert to lowercase
    data['processed'] = data['Review'].apply(lambda x: ' '.join([
        lemmatizer.lemmatize(word.lower())  # Lemmatize and convert to lowercase
        for word in word_tokenize(str(x))  # Use word_tokenize for robust tokenization
        if word.lower() not in stop_words  # Remove stopwords
    ]))
    
    return data