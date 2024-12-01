from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from softmax import softmax
from entropy import calculate_entropy, compute_dynamic_threshold

# Load RoBERTa tokenizer and model (only once at the beginning)
if 'tokenizer' not in globals() or 'model' not in globals():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

def process_reviews(reviews):
    """
    Processes reviews to calculate sentiment scores and filter uncertain words based on entropy.
    """
    filtered_reviews = []  # Store filtered reviews
    entropy_values = []  # Store entropy for each word
    sentiment_scores = []  # Store sentiment scores for reviews (list of [positive, neutral, negative])
    
    for review in reviews:
        # Tokenize the review
        tokens = tokenizer(review, return_tensors="pt", padding=True, truncation=True)
        
        # Get logits from the model
        with torch.no_grad():
            outputs = model(**tokens)
        logits = outputs.logits.squeeze().numpy()
        
        # Convert logits to probabilities
        probabilities = softmax(logits)
        
        # Append sentiment probabilities (positive, neutral, negative) to sentiment_scores
        sentiment_scores.append(probabilities.tolist())  # Ensure it's a list of lists
        
        # Debugging output for probabilities
        print(f"Review: {review}")
        print(f"Logits: {logits}")
        print(f"Probabilities: {probabilities}")

        # Check if any probability is significantly low
        if np.any(probabilities < 0.05):  # Adjust threshold as necessary
            print("Warning: Low probability detected.")

        # Calculate entropy for each review
        ent = calculate_entropy(probabilities)
        entropy_values.append(ent)
        
        # Filter based on entropy threshold
        threshold = compute_dynamic_threshold(entropy_values)
        if ent <= threshold:  # Retain if entropy is below the threshold
            filtered_reviews.append(review)

    return filtered_reviews, entropy_values, sentiment_scores
