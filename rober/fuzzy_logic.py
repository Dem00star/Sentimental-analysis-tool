import pandas as pd
import numpy as np
import skfuzzy as fuzz
from aggregate_score import mean_score
import warnings
from numpy import ma

# Filter RuntimeWarnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

class FuzzySentimentConverter:
    def __init__(self, sentiment_range=(-1, 1), rating_range=(0, 5)):
        """Initialize fuzzy sentiment conversion with configurable ranges"""
        self.sentiment_min, self.sentiment_max = sentiment_range
        self.rating_min, self.rating_max = rating_range
        self.sentiment = np.arange(self.sentiment_min, self.sentiment_max + 0.1, 0.1)
        self.rating = np.arange(self.rating_min, self.rating_max + 0.1, 0.1)
        self.sentiment_mid = 0
        self.rating_mid = 2.5

    def safe_defuzz(self, x, mf, method='centroid'):
        """Safely perform defuzzification with error handling"""
        try:
            # Use masked arrays to handle invalid values
            masked_mf = ma.masked_invalid(mf)
            if masked_mf.mask.all():  # If all values are invalid
                return self.rating_mid
            return fuzz.defuzz(x, mf, method)
        except Exception as e:
            print(f"Defuzzification error: {str(e)}")
            return self.rating_mid

    def create_membership_functions(self):
        """Create membership functions with proper error handling"""
        return {
            'Gaussian': {
                'sentiment': {
                    'low': fuzz.gaussmf(self.sentiment, -0.5, 0.3),
                    'neutral': fuzz.gaussmf(self.sentiment, 0, 0.3),
                    'high': fuzz.gaussmf(self.sentiment, 0.5, 0.3)
                },
                'rating': {
                    'low': fuzz.gaussmf(self.rating, 1, 1),
                    'neutral': fuzz.gaussmf(self.rating, 2.5, 1),
                    'high': fuzz.gaussmf(self.rating, 4, 1)
                }
            },
            'Sigmoidal': {
                'sentiment': {
                    'low': fuzz.sigmf(self.sentiment, -0.5, -10),
                    'neutral': fuzz.psigmf(self.sentiment, -0.2, 10, 0.2, -10),
                    'high': fuzz.sigmf(self.sentiment, 0.5, 10)
                },
                'rating': {
                    'low': fuzz.sigmf(self.rating, 1, -10),
                    'neutral': fuzz.psigmf(self.rating, 1.5, 10, 3.5, -10),
                    'high': fuzz.sigmf(self.rating, 4, 10)
                }
            },
            'Triangular': {
                'sentiment': {
                    'low': fuzz.trimf(self.sentiment, [-1, -0.5, 0]),
                    'neutral': fuzz.trimf(self.sentiment, [-0.5, 0, 0.5]),
                    'high': fuzz.trimf(self.sentiment, [0, 0.5, 1])
                },
                'rating': {
                    'low': fuzz.trimf(self.rating, [0, 1, 2]),
                    'neutral': fuzz.trimf(self.rating, [2, 2.5, 3]),
                    'high': fuzz.trimf(self.rating, [3, 4, 5])
                }
            },
            'Trapezoidal': {
                'sentiment': {
                    'low': fuzz.trapmf(self.sentiment, [-1, -1, -0.3, 0]),
                    'neutral': fuzz.trapmf(self.sentiment, [-0.3, -0.1, 0.1, 0.3]),
                    'high': fuzz.trapmf(self.sentiment, [0, 0.3, 1, 1])
                },
                'rating': {
                    'low': fuzz.trapmf(self.rating, [0, 0, 1, 2]),
                    'neutral': fuzz.trapmf(self.rating, [1.5, 2, 3, 3.5]),
                    'high': fuzz.trapmf(self.rating, [3, 4, 5, 5])
                }
            },
            'Piecewise': {
                'sentiment': {
                    'low': fuzz.piecemf(self.sentiment, [-1, -0.5, 0]),
                    'neutral': fuzz.piecemf(self.sentiment, [-0.5, 0, 0.5]),
                    'high': fuzz.piecemf(self.sentiment, [0, 0.5, 1])
                },
                'rating': {
                    'low': fuzz.piecemf(self.rating, [0, 1.25, 2.5]),
                    'neutral': fuzz.piecemf(self.rating, [1.25, 2.5, 3.75]),
                    'high': fuzz.piecemf(self.rating, [2.5, 3.75, 5])
                }
            }
        }

    def fuzzify_sentiment_score(self, sentiment_score, membership_type):
        """Convert sentiment score to rating with robust error handling"""
        try:
            # Input validation
            if not isinstance(sentiment_score, (int, float)):
                raise ValueError(f"Invalid sentiment score type: {type(sentiment_score)}")
            
            if not (self.sentiment_min <= sentiment_score <= self.sentiment_max):
                raise ValueError(f"Sentiment score {sentiment_score} out of range")

            memberships = self.create_membership_functions()
            membership = memberships[membership_type]
            
            # Compute membership levels with error handling
            low_level = max(0, min(1, fuzz.interp_membership(
                self.sentiment, 
                membership['sentiment']['low'], 
                sentiment_score
            )))
            neutral_level = max(0, min(1, fuzz.interp_membership(
                self.sentiment, 
                membership['sentiment']['neutral'], 
                sentiment_score
            )))
            high_level = max(0, min(1, fuzz.interp_membership(
                self.sentiment, 
                membership['sentiment']['high'], 
                sentiment_score
            )))
            
            # Calculate weighted average rating with safety checks
            total_membership = low_level + neutral_level + high_level
            
            if total_membership > 0:
                low_rating = self.safe_defuzz(self.rating, membership['rating']['low'])
                neutral_rating = self.safe_defuzz(self.rating, membership['rating']['neutral'])
                high_rating = self.safe_defuzz(self.rating, membership['rating']['high'])
                
                rating_value = (
                    low_level * low_rating +
                    neutral_level * neutral_rating +
                    high_level * high_rating
                ) / total_membership
            else:
                rating_value = self.rating_mid
            
            return np.clip(rating_value, self.rating_min, self.rating_max)
            
        except Exception as e:
            print(f"Error in fuzzification ({membership_type}): {str(e)}")
            return self.rating_mid

def process_and_save():
    """Process sentiment scores and save results"""
    try:
        # Get DataFrame from mean_score
        df = mean_score()
        
        # Initialize fuzzy converter
        converter = FuzzySentimentConverter()
        
        # Process each membership function
        membership_types = ['Gaussian', 'Sigmoidal', 'Triangular', 'Trapezoidal', 'Piecewise']
        
        for mtype in membership_types:
            column_name = f'Rating_{mtype}'
            df[column_name] = df['mean_filtered_sentiment_score'].apply(
                lambda x: converter.fuzzify_sentiment_score(x, mtype)
            )
        
        # Select and reorder columns
        columns = ['CourseId', 'mean_filtered_sentiment_score'] + \
                 [f'Rating_{mtype}' for mtype in membership_types] + \
                 ['mean_label']
        
        output_df = df[columns]
        
        # Save to CSV with proper formatting
        output_df.to_csv('fuzzy_ratings.csv', index=False, float_format='%.4f')
        print("Results saved to fuzzy_ratings.csv")
        print(output_df)

        
        # Display first few rows
        print("\nFirst few rows of processed data:")
        print(output_df.head().to_string())
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    process_and_save()