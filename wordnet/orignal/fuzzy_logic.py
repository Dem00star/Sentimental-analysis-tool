import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class Type2FuzzyLogicSystem:
    def __init__(self):
        self.rating_range = np.linspace(1, 5, 100)
        self.sentiment_range = np.linspace(-1, 1, 100)
        self.default_std = 0.5  # Default standard deviation for empty/invalid cases

    def validate_input_arrays(self, sentiment_scores, entropy_values):
        """Validate input arrays and handle empty cases"""
        if not isinstance(sentiment_scores, (list, np.ndarray)) or not isinstance(entropy_values, (list, np.ndarray)):
            raise ValueError("sentiment_scores and entropy_values must be lists or numpy arrays")
        
        if len(sentiment_scores) == 0 or len(entropy_values) == 0:
            return False
        
        return True

    def safe_mean(self, arr):
        """Safely calculate mean of an array"""
        if len(arr) == 0:
            return 0.0
        return float(np.mean(arr))

    def safe_std(self, arr):
        """Safely calculate standard deviation of an array"""
        if len(arr) <= 1:
            return self.default_std
        return float(np.std(arr))

    def gaussian_mf(self, x, mean, std):
        """Create a Gaussian membership function with safety checks"""
        if std <= 0:
            std = self.default_std
        return np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    def create_fou(self, mean_sentiment, mean_entropy, std_entropy):
        """Create Footprint of Uncertainty using entropy statistics with bounds checking"""
        # Ensure values are within expected ranges
        mean_sentiment = np.clip(mean_sentiment, -1, 1)
        mean_entropy = np.clip(mean_entropy, 0, 1)
        std_entropy = np.clip(std_entropy, 0, 1)

        # Define membership parameters based on entropy
        lower_std = max(0.3, 1 - mean_entropy)
        upper_std = min(2.0, 1 + mean_entropy)
        
        membership_functions = {
            'low': {
                'mean': 1,
                'lower_std': lower_std,
                'upper_std': upper_std
            },
            'medium': {
                'mean': 2.5,
                'lower_std': lower_std,
                'upper_std': upper_std
            },
            'high': {
                'mean': 4,
                'lower_std': lower_std,
                'upper_std': upper_std
            },
            'very_high': {
                'mean': 5,
                'lower_std': lower_std * 0.8,
                'upper_std': upper_std * 0.8
            }
        }
        
        return membership_functions

    def calculate_rating(self, sentiment_scores, entropy_values):
        """Calculate rating using Type-2 Fuzzy Logic with input validation"""
        try:
            # Validate inputs
            if not self.validate_input_arrays(sentiment_scores, entropy_values):
                return 3.0, np.zeros_like(self.rating_range), np.zeros_like(self.rating_range)

            # Convert inputs to numpy arrays if they aren't already
            sentiment_scores = np.array(sentiment_scores)
            entropy_values = np.array(entropy_values)

            mean_sentiment = self.safe_mean(sentiment_scores)
            mean_entropy = self.safe_mean(entropy_values)
            std_entropy = self.safe_std(entropy_values)

            # Get FOU parameters
            membership_functions = self.create_fou(mean_sentiment, mean_entropy, std_entropy)
            
            # Initialize membership arrays
            lower_memberships = np.zeros_like(self.rating_range)
            upper_memberships = np.zeros_like(self.rating_range)
            
            # Map sentiment to rating scale
            mapped_sentiment = 1 + (mean_sentiment + 1) * 2
            
            # Calculate memberships for each fuzzy set
            for fuzzy_set, params in membership_functions.items():
                lower_mf = self.gaussian_mf(self.rating_range, params['mean'], params['lower_std'])
                upper_mf = self.gaussian_mf(self.rating_range, params['mean'], params['upper_std'])
                
                weight = np.exp(-abs(params['mean'] - mapped_sentiment))
                lower_memberships += lower_mf * weight
                upper_memberships += upper_mf * weight
            
            # Avoid division by zero
            max_lower = np.max(lower_memberships)
            max_upper = np.max(upper_memberships)
            
            if max_lower > 0:
                lower_memberships = lower_memberships / max_lower
            if max_upper > 0:
                upper_memberships = upper_memberships / max_upper
            
            # Calculate centroid with safety checks
            denominator = np.sum(lower_memberships + upper_memberships)
            if denominator > 0:
                rating = np.sum(self.rating_range * (lower_memberships + upper_memberships)) / denominator
            else:
                rating = 3.0  # Default middle rating
            
            return np.clip(rating, 1, 5), lower_memberships, upper_memberships
            
        except Exception as e:
            print(f"Error in calculate_rating: {str(e)}")
            return 3.0, np.zeros_like(self.rating_range), np.zeros_like(self.rating_range)

    def visualize_membership(self, review_idx, sentiment_scores, entropy_values, rating):
        """Visualize the Type-2 fuzzy membership functions with error handling"""
        try:
            if review_idx >= 10:
                return
                
            plt.figure(figsize=(12, 6))
            mean_sentiment = self.safe_mean(sentiment_scores)
            mean_entropy = self.safe_mean(entropy_values)
            
            # Calculate membership functions
            rating, lower_mf, upper_mf = self.calculate_rating(sentiment_scores, entropy_values)
            
            # Plot
            plt.plot(self.rating_range, lower_mf, '--b', label='Lower MF')
            plt.plot(self.rating_range, upper_mf, '-b', label='Upper MF')
            plt.fill_between(self.rating_range, lower_mf, upper_mf, alpha=0.3, 
                            label='Footprint of Uncertainty')
            plt.axvline(x=rating, color='r', linestyle='--', 
                       label=f'Rating: {rating:.2f}')
            
            plt.title(f'Review #{review_idx+1}\n' + 
                     f'Mean Sentiment: {mean_sentiment:.2f}, Mean Entropy: {mean_entropy:.2f}')
            plt.xlabel('Rating')
            plt.ylabel('Membership Degree')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"Error in visualize_membership: {str(e)}")

def main():
    try:
        # Read data
        df = pd.read_csv('c.csv')
        
        # Safely convert string representations to lists
        def safe_eval(x):
            try:
                return eval(x) if isinstance(x, str) else x
            except:
                return []

        df['sentiment_scores'] = df['sentiment_scores'].apply(safe_eval)
        df['entropy_values'] = df['entropy_values'].apply(safe_eval)
        
        fuzzy_system = Type2FuzzyLogicSystem()
        ratings = []
        
        # Process each review
        for idx, row in df.iterrows():
            rating, lower_mf, upper_mf = fuzzy_system.calculate_rating(
                row['sentiment_scores'],
                row['entropy_values']
            )
            ratings.append(rating)
            
            # Visualize first 10 reviews
            fuzzy_system.visualize_membership(
                idx,
                row['sentiment_scores'],
                row['entropy_values'],
                rating
            )
        
        # Save results
        df['fuzzy_rating'] = ratings
        df.to_csv('d.csv', index=False)
        
    except Exception as e:
        print(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()