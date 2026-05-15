import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('rating_ama.csv')

# Convert 'label' to numeric, dropping invalid entries
df['label_numeric'] = pd.to_numeric(df['label'], errors='coerce')  # Invalid entries -> NaN
df = df.dropna(subset=['label_numeric'])  # Drop rows with NaN in 'label_numeric'
df['label_numeric'] = df['label_numeric'].astype(int)  # Ensure labels are integers

# Normalize and binarize 'label_numeric'
df['normalized_label'] = (df['label_numeric'] - df['label_numeric'].min()) / (df['label_numeric'].max() - df['label_numeric'].min())
df['processed_label'] = df['normalized_label'].apply(lambda x: 0 if x < 0.5 else 1)

# Normalize and binarize 'fuzzy_rating'
df['normalized_fuzzy_rating'] = (df['fuzzy_rating'] - 1) / 4  # Normalize to [0, 1]
df['binary_fuzzy_rating'] = df['normalized_fuzzy_rating'].apply(lambda x: 0 if x < 0.5 else 1)

# Ensure both columns have valid values
print("Unique values in processed_label:", df['processed_label'].unique())
print("Unique values in binary_fuzzy_rating:", df['binary_fuzzy_rating'].unique())

# Calculate precision, recall, and F1 score
if len(df['processed_label'].unique()) > 1 and len(df['binary_fuzzy_rating'].unique()) > 1:
    precision = precision_score(df['processed_label'], df['binary_fuzzy_rating'], zero_division=0)
    recall = recall_score(df['processed_label'], df['binary_fuzzy_rating'], zero_division=0)
    f1 = f1_score(df['processed_label'], df['binary_fuzzy_rating'], zero_division=0)

    # Print the results
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
else:
    print("Error: Either processed_label or binary_fuzzy_rating contains only one class.")
