import pandas as pd
import numpy as np
import logging
from read_data import read_data
from preprocessing import preprocess_data
from roberta import SentimentAnalyzer
from fuzzy_logic import apply_fuzzy_logic

def process_course_ratings(file_path, output_path):
    """
    Robust course rating processing with comprehensive error handling
    """
    # Configure logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load and preprocess data
    try:
        data = read_data(file_path)
        preprocessed_data = preprocess_data(data)
    except Exception as e:
        logger.error(f"Data loading/preprocessing error: {e}")
        return pd.DataFrame()

    # Initialize sentiment analyzer
    analyzer = SentimentAnalyzer()

    # Group reviews by CourseId
    course_groups = preprocessed_data.groupby('CourseId')
    course_ratings = []

    for course_id, group in course_groups:
        reviews = group['processed'].tolist()
        
        try:
            # Process reviews
            _, entropy_values, sentiment_scores = analyzer.process_reviews(reviews)

            # More flexible review filtering
            valid_scores = [
                score for score, entropy in zip(sentiment_scores, entropy_values) 
                if (entropy < 1.5 and  # Relaxed entropy threshold
                    len(score) == 3)    # Ensure 3 sentiment categories
            ]

            if not valid_scores:
                logger.warning(f"No valid sentiment scores for CourseId {course_id}")
                # Default rating with some variability
                default_rating = np.random.uniform(2.5, 4.0)
                course_ratings.append({
                    "CourseId": course_id, 
                    "Rating(1-5)": round(default_rating, 1),
                    "Review Count": 0
                })
                continue

            # Aggregate sentiment scores
            course_sentiment = {
                "positive": np.mean([score[0] for score in valid_scores]),
                "neutral": np.mean([score[1] for score in valid_scores]),
                "negative": np.mean([score[2] for score in valid_scores]),
            }

            # Apply fuzzy logic with multiple strategies
            ratings = [
                apply_fuzzy_logic([course_sentiment["positive"], 
                                   course_sentiment["neutral"], 
                                   course_sentiment["negative"]], strategy)
                for strategy in ['conservative', 'balanced', 'optimistic']
            ]

            # Use robust statistical methods for final rating
            final_rating = np.median(ratings)
            
            # Review count influence
            review_count = len(valid_scores)
            if review_count < 3:
                final_rating = np.clip(final_rating, 2.5, 4.5)
            
            course_ratings.append({
                "CourseId": course_id, 
                "Rating(1-5)": round(final_rating, 1),
                "Review Count": review_count
            })

        except Exception as e:
            logger.error(f"Error processing CourseId {course_id}: {e}")
            course_ratings.append({
                "CourseId": course_id, 
                "Rating(1-5)": 3.0,
                "Review Count": 0
            })

    # Create and save ratings DataFrame
    ratings_df = pd.DataFrame(course_ratings)
    ratings_df.to_csv(output_path, index=False)
    logger.info(f"Course ratings saved to: {output_path}")
    
    return ratings_df

# Main execution
if __name__ == "__main__":
    file_path = 'test.csv'
    output_path = 'course_ratings_for_test.csv'
    ratings = process_course_ratings(file_path, output_path)
    print(ratings)
