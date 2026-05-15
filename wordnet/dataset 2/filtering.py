import pandas as pd
import ast

def filter_reviews(input_file, threshold, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    filtered_data = []
    
    for _, row in df.iterrows():
        try:
            # Convert string representations to Python lists
            tokens = ast.literal_eval(row['tokens'])
            sentiments = ast.literal_eval(row['sentiment_scores'])
            entropies = ast.literal_eval(row['entropy_values'])
            
            # Filter based on entropy threshold
            filtered_indices = [i for i, entropy in enumerate(entropies) if entropy <= threshold]
            
            # Skip if all tokens have entropy > threshold
            if not filtered_indices:
                continue
                
            # Create filtered lists
            filtered_tokens = [tokens[i] for i in filtered_indices]
            filtered_sentiments = [sentiments[i] for i in filtered_indices]
            filtered_entropies = [entropies[i] for i in filtered_indices]
            
            # Create new row with filtered data
            filtered_row = {
                'course': row['course'],
                'reviews': row['reviews'],
                'tokens': str(filtered_tokens),
                'sentiment_scores': str(filtered_sentiments),
                'label': row['label'],
                'entropy_values': str(filtered_entropies)
            }
            
            filtered_data.append(filtered_row)
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    # Create new dataframe and save to CSV
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df.to_csv(output_file, index=False)
    
    # Print statistics
    print(f"\nFiltering Statistics:")
    print(f"Original reviews: {len(df)}")
    print(f"Filtered reviews: {len(filtered_df)}")
    print(f"Reviews removed: {len(df) - len(filtered_df)}")

if __name__ == "__main__":
    INPUT_FILE = "entropy_ama.csv"
    OUTPUT_FILE = "am.csv"
    THRESHOLD = 0.975  # Example threshold
    
    filter_reviews(INPUT_FILE, THRESHOLD, OUTPUT_FILE)





