import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from aggregate_score import mean_score

class Type2FuzzySentimentConverter:
    def __init__(self, sentiment_range=(0, 1), rating_range=(0, 5)):
        self.sentiment_min, self.sentiment_max = sentiment_range
        self.rating_min, self.rating_max = rating_range
        self.sentiment = np.arange(self.sentiment_min, self.sentiment_max + 0.01, 0.01)
        self.rating = np.arange(self.rating_min, self.rating_max + 0.1, 0.1)
        self.rating_mid = 2.5

    def create_membership_functions(self):
        def fou(lower, upper):
            return {'lower': lower, 'upper': upper}

        return {
            'Gaussian': {
                'sentiment': {
                    'low': fou(
                        fuzz.gaussmf(self.sentiment, 0.2, 0.1),
                        fuzz.gaussmf(self.sentiment, 0.2, 0.15)
                    ),
                    'medium': fou(
                        fuzz.gaussmf(self.sentiment, 0.4, 0.1),
                        fuzz.gaussmf(self.sentiment, 0.4, 0.15)
                    ),
                    'high': fou(
                        fuzz.gaussmf(self.sentiment, 0.6, 0.1),
                        fuzz.gaussmf(self.sentiment, 0.6, 0.15)
                    ),
                    'very_high': fou(
                        fuzz.gaussmf(self.sentiment, 0.8, 0.1),
                        fuzz.gaussmf(self.sentiment, 0.8, 0.15)
                    )
                },
                'rating': {
                    'low': fou(
                        fuzz.gaussmf(self.rating, 3, 0.5),
                        fuzz.gaussmf(self.rating, 3, 0.8)
                    ),
                    'medium': fou(
                        fuzz.gaussmf(self.rating, 4, 0.5),
                        fuzz.gaussmf(self.rating, 4, 0.8)
                    ),
                    'high': fou(
                        fuzz.gaussmf(self.rating, 4.5, 0.5),
                        fuzz.gaussmf(self.rating, 4.5, 0.8)
                    ),
                    'very_high': fou(
                        fuzz.gaussmf(self.rating, 5, 0.3),
                        fuzz.gaussmf(self.rating, 5, 0.5)
                    )
                }
            }
        }

    def fuzzify_sentiment_score(self, sentiment_score, membership_type):
        try:
            if not isinstance(sentiment_score, (int, float)):
                raise ValueError(f"Invalid sentiment score type: {type(sentiment_score)}")

            memberships = self.create_membership_functions()
            membership = memberships[membership_type]

            def compute_level(fou):
                lower_level = fuzz.interp_membership(self.sentiment, fou['lower'], sentiment_score)
                upper_level = fuzz.interp_membership(self.sentiment, fou['upper'], sentiment_score)
                return lower_level, upper_level

            levels = {
                category: compute_level(membership['sentiment'][category])
                for category in ['low', 'medium', 'high', 'very_high']
            }

            lower_rating_mf = sum(
                levels[cat][0] * membership['rating'][cat]['lower']
                for cat in levels
            )
            upper_rating_mf = sum(
                levels[cat][1] * membership['rating'][cat]['upper']
                for cat in levels
            )

            rating_value = self.safe_defuzz_type2(self.rating, lower_rating_mf, upper_rating_mf)
            return np.clip(rating_value * 1.1, self.rating_min, self.rating_max)

        except Exception as e:
            print(f"Error in fuzzification ({membership_type}): {str(e)}")
            return self.rating_mid

    def safe_defuzz_type2(self, x, lower_mf, upper_mf, method='centroid'):
        try:
            lower_defuzz = fuzz.defuzz(x, lower_mf, method)
            upper_defuzz = fuzz.defuzz(x, upper_mf, method)
            return (lower_defuzz + upper_defuzz) / 2
        except Exception as e:
            print(f"Defuzzification error: {str(e)}")
            return self.rating_mid

def plot_membership_functions(converter):
    plt.figure(figsize=(12, 8))
    
    memberships = converter.create_membership_functions()['Gaussian']['sentiment']
    categories = ['low', 'medium', 'high', 'very_high']
    colors = ['r', 'g', 'b', 'purple']
    styles = ['-', '--']
    
    for cat, color in zip(categories, colors):
        plt.plot(converter.sentiment, memberships[cat]['lower'], 
                color=color, linestyle='--', label=f"{cat} (Lower)")
        plt.plot(converter.sentiment, memberships[cat]['upper'], 
                color=color, linestyle='-', label=f"{cat} (Upper)")
    
    plt.title("Adjusted Type-2 Fuzzy Membership Functions")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True)
    plt.show()

def process_and_save():
    try:
        df = mean_score()
        converter = Type2FuzzySentimentConverter()
        
        df['Rating_Gaussian_Type2'] = df['mean_filtered_sentiment_score'].apply(
            lambda x: converter.fuzzify_sentiment_score(x, 'Gaussian')
        )
        
        output_df = df[['CourseId', 'mean_filtered_sentiment_score', 
                       'Rating_Gaussian_Type2', 'mean_label']]
        
        output_df.to_csv('type2_fuzzy_ratings_adjusted.csv', index=False, float_format='%.4f')
        print("Results saved to type2_fuzzy_ratings_adjusted.csv")
        print("\nFirst few rows:")
        print(output_df.head().to_string())
        
        plot_membership_functions(converter)
        
    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    process_and_save()