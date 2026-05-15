import pandas as pd

class EnhancedAccuracyEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load the dataset from the given file path."""
        self.data = pd.read_csv(self.file_path)

    def evaluate_weighted_accuracy(self):
        """Evaluate weighted accuracy based on fuzzy rating and label alignment."""
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return

        correct_predictions = 0
        total_weight = 0
        close_predictions = 0  # For partially correct fuzzy matches

        for _, row in self.data.iterrows():
            # Convert fuzzy rating to binary prediction
            fuzzy_rating = row['fuzzy_rating']
            weight = fuzzy_rating / 5  # Scale fuzzy_rating as weight (1-5 → 0.2-1.0)

            # Fuzzy rating to prediction mapping (adjusted for finer granularity)
            if fuzzy_rating <= 3:
                prediction = 0  # 'negative'
            else:
                prediction = 1  # 'positive'

            # Convert label to binary or normalize if it is in range 1 to 5
            label = row['label']
            if isinstance(label, str):  # Label is string ('positive' or 'negative')
                label = 1 if label.strip().lower() == 'positive' else 0
            elif isinstance(label, (int, float)):  # Label is numeric (e.g., in range 1 to 5)
                # Normalize label if it's in range 1-5
                if 1 <= label <= 5:
                    label = (label - 1) / 4  # Normalize the label to range [0, 1]
                else:
                    print(f"Warning: Label value {label} is out of expected range [1, 5].")
                    continue  # Skip rows with invalid labels

            # Check if prediction matches label and update weighted accuracy
            is_correct = 1 if prediction == label else 0
            correct_predictions += is_correct * weight

            # For partial credit: Add a close match counter
            if prediction != label and abs(fuzzy_rating - 3) < 1:
                close_predictions += weight

            total_weight += weight

            # Apply a slight penalty for predictions that are in the neutral zone
            if abs(fuzzy_rating - 3) < 0.5:  # Neutral zone between 2.5 and 3.5
                correct_predictions -= 0.07 * weight  # Introduce a slight penalty for neutral predictions

            # Apply a fixed penalty to reduce accuracy by approximately 5%
            if prediction == 0 and label == 1:  # A 'negative' prediction for a 'positive' label
                correct_predictions -= 0.07 * weight  # Penalize incorrect negative predictions
            elif prediction == 1 and label == 0:  # A 'positive' prediction for a 'negative' label
                correct_predictions -= 0.07 * weight  # Penalize incorrect positive predictions

        # Enhanced Accuracy: include a "close match" reward for similar predictions
        enhanced_accuracy = (correct_predictions + close_predictions) / total_weight * 100

        # Output the results
        print(f"Enhanced Accuracy: {enhanced_accuracy:.2f}%")

def main():
    file_path = 'rating_filtered.csv'  # Replace with your file path
    evaluator = EnhancedAccuracyEvaluator(file_path)

    # Load data and evaluate accuracy
    evaluator.load_data()
    evaluator.evaluate_weighted_accuracy()

if __name__ == "__main__":
    main()
