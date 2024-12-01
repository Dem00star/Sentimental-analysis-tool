import pandas as pd
from read_data import read_data
from preprocessing import preprocess_data
from roberta import process_reviews
from fuzzy_logic import apply_fuzzy_logic


# File paths
file_path = '/Users/rohan/Documents/paper/test.csv'
output_path = '/Users/rohan/Documents/paper/course_ratings.csv'

# Load the CSV file
data = read_data(file_path)

# Preprocess the reviews
preprocessed_data = preprocess_data(data)

# Extract reviews grouped by CourseId
course_groups = preprocessed_data.groupby('CourseId')

# Create a list to store course ratings
course_ratings = []

# Process each course group
for course_id, group in course_groups:
    # Extract processed reviews for the course
    reviews = group['processed'].tolist()
    
    # Process the reviews: calculate entropy, filter uncertain words, and compute sentiment scores
    _, _, sentiment_scores = process_reviews(reviews)
    
    # Aggregate sentiment scores for the course
    course_sentiment = {
        "positive": sum(score[0] for score in sentiment_scores) / len(sentiment_scores),
        "neutral": sum(score[1] for score in sentiment_scores) / len(sentiment_scores),
        "negative": sum(score[2] for score in sentiment_scores) / len(sentiment_scores),
    }
    
    # Apply fuzzy logic to determine the course rating based on sentiment scores
    course_rating = apply_fuzzy_logic([course_sentiment["positive"], course_sentiment["neutral"], course_sentiment["negative"]])
    
    # Round rating to one decimal place
    course_ratings.append({"CourseId": course_id, "Rating(1-5)": round(course_rating, 1)})

# Create a DataFrame from the course ratings
ratings_df = pd.DataFrame(course_ratings)

# Save the DataFrame to a new CSV file
ratings_df.to_csv(output_path, index=False)

print(f"Course ratings saved to: {output_path}")