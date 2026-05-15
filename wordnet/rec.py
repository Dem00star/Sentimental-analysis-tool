import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('r_im_filtered.csv')

# Handle 'label' based on its type
if set(df['label'].unique()) == {'positive', 'negative'}:  # If label contains 'positive' or 'negative'
    df['processed_label'] = df['label'].apply(lambda x: 1 if x == 'positive' else 0)
elif set(df['label'].unique()).issubset({0, 1}):  # If label is already binary (0 or 1)
    df['processed_label'] = df['label']
else:
    print("Error: Label contains unexpected values.")
    exit()

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
