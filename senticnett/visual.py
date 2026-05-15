import pandas as pd
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from aggregate_score import mean_score  # Ensure this module is available

class Type2FuzzySentimentConverter:
    def __init__(self, sentiment_range=(-1, 1), rating_range=(0, 5)):
        """Initialize fuzzy sentiment conversion with configurable ranges"""
        self.sentiment_min, self.sentiment_max = sentiment_range
        self.rating_min, self.rating_max = rating_range
        self.sentiment = np.arange(self.sentiment_min, self.sentiment_max + 0.1, 0.1)
        self.rating = np.arange(self.rating_min, self.rating_max + 0.1, 0.1)
        self.rating_mid = 2.5

    def safe_defuzz_type2(self, x, lower_mf, upper_mf, method='centroid'):
        """Safely perform defuzzification for Type-2 fuzzy sets"""
        try:
            lower_defuzz = fuzz.defuzz(x, lower_mf, method)
            upper_defuzz = fuzz.defuzz(x, upper_mf, method)
            return (lower_defuzz + upper_defuzz) / 2
        except Exception as e:
            print(f"Defuzzification error: {str(e)}")
            return self.rating_mid

    def create_membership_functions(self):
        """Create neutral and symmetric Type-2 membership functions (FOU)"""
        def fou(lower, upper):
            """Helper to generate lower and upper FOU bounds"""
            return {'lower': lower, 'upper': upper}

        return {
            'Gaussian': {
                'sentiment': {
                    'low': fou(
                        fuzz.gaussmf(self.sentiment, -0.5, 0.2),
                        fuzz.gaussmf(self.sentiment, -0.5, 0.3)
                    ),
                    'neutral': fou(
                        fuzz.gaussmf(self.sentiment, 0, 0.2),
                        fuzz.gaussmf(self.sentiment, 0, 0.3)
                    ),
                    'high': fou(
                        fuzz.gaussmf(self.sentiment, 0.5, 0.2),
                        fuzz.gaussmf(self.sentiment, 0.5, 0.3)
                    )
                },
                'rating': {
                    'low': fou(
                        fuzz.gaussmf(self.rating, 1.5, 0.3),
                        fuzz.gaussmf(self.rating, 1.5, 0.5)
                    ),
                    'neutral': fou(
                        fuzz.gaussmf(self.rating, 2.5, 0.3),
                        fuzz.gaussmf(self.rating, 2.5, 0.5)
                    ),
                    'high': fou(
                        fuzz.gaussmf(self.rating, 4, 0.3),
                        fuzz.gaussmf(self.rating, 4, 0.5)
                    )
                }
            }
        }

    def fuzzify_sentiment_score(self, sentiment_score, membership_type):
        """Convert sentiment score to rating using Type-2 fuzzy logic"""
        try:
            if not isinstance(sentiment_score, (int, float)):
                raise ValueError(f"Invalid sentiment score type: {type(sentiment_score)}")

            memberships = self.create_membership_functions()
            membership = memberships[membership_type]

            def compute_level(fou):
                lower_level = fuzz.interp_membership(self.sentiment, fou['lower'], sentiment_score)
                upper_level = fuzz.interp_membership(self.sentiment, fou['upper'], sentiment_score)
                return lower_level, upper_level

            low_level = compute_level(membership['sentiment']['low'])
            neutral_level = compute_level(membership['sentiment']['neutral'])
            high_level = compute_level(membership['sentiment']['high'])

            lower_rating_mf = (
                low_level[0] * membership['rating']['low']['lower'] +
                neutral_level[0] * membership['rating']['neutral']['lower'] +
                high_level[0] * membership['rating']['high']['lower']
            )
            upper_rating_mf = (
                low_level[1] * membership['rating']['low']['upper'] +
                neutral_level[1] * membership['rating']['neutral']['upper'] +
                high_level[1] * membership['rating']['high']['upper']
            )

            rating_value = self.safe_defuzz_type2(self.rating, lower_rating_mf, upper_rating_mf)
            return np.clip(rating_value, self.rating_min, self.rating_max)

        except Exception as e:
            print(f"Error in fuzzification ({membership_type}): {str(e)}")
            return self.rating_mid

def plot_membership_functions(x, membership_fns, title):
    """Plot membership functions"""
    plt.figure(figsize=(10, 6))
    for label, fou in membership_fns.items():
        plt.plot(x, fou['lower'], label=f"{label.capitalize()} (Lower)", linestyle="--")
        plt.plot(x, fou['upper'], label=f"{label.capitalize()} (Upper)", linestyle="-")
    plt.title(title)
    plt.xlabel("Domain")
    plt.ylabel("Membership Degree")
    plt.legend()
    plt.grid(True)
    plt.show()

def process_and_save():
    """Process sentiment scores and save results using Type-2 fuzzy logic"""
    try:
        df = mean_score()
        converter = Type2FuzzySentimentConverter()
        membership_types = ['Gaussian']

        for mtype in membership_types:
            column_name = f'Rating_{mtype}_Type2'
            df[column_name] = df['mean_filtered_sentiment_score'].apply(
                lambda x: converter.fuzzify_sentiment_score(x, mtype)
            )

        columns = ['CourseId', 'mean_filtered_sentiment_score'] + \
                  [f'Rating_{mtype}_Type2' for mtype in membership_types] + \
                  ['mean_label']

        output_df = df[columns]
        output_df.to_csv('type2_fuzzy_ratings.csv', index=False, float_format='%.4f')
        print("Results saved to type2_fuzzy_ratings.csv")

        print("\nFirst few rows of processed data:")
        print(output_df.head().to_string())

        # Visualize Sentiment Membership Functions
        memberships = converter.create_membership_functions()['Gaussian']
        plot_membership_functions(converter.sentiment, memberships['sentiment'], "Sentiment Membership Functions")
        plot_membership_functions(converter.rating, memberships['rating'], "Rating Membership Functions")

    except Exception as e:
        print(f"Error in processing: {str(e)}")

if __name__ == "__main__":
    process_and_save()
