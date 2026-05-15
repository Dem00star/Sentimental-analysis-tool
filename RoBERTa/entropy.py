import pandas as pd
import numpy as np
from collections import Counter
from my_roberta import roberta_score  # Assuming roberta_score is defined elsewhere
import csv

# Function to calculate Shannon entropy
def shannon_entropy(tokens):
    # Calculate frequency of each token
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    
    entropy = 0
    for count in token_counts.values():
        # Calculate the probability of each token
        prob = count / total_tokens
        entropy -= prob * np.log2(prob)
    
    return entropy

# Function to read labels from test.csv and calculate the mean Label for each CourseId
def calculate_mean_labels(file_path):
    labels_dict = {}
    
    # Read the CSV file
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            course_id = row['CourseId']
            label = float(row['Label'])
            
            if course_id not in labels_dict:
                labels_dict[course_id] = []
            labels_dict[course_id].append(label)
    
    # Calculate the mean label for each course
    for course_id in labels_dict:
        labels_dict[course_id] = np.mean(labels_dict[course_id])
    
    return labels_dict

# Function to process the dataframe, calculate entropy, and filter based on threshold
def calculate_filtered_scores(label_file_path):
    # Get the dataframe from roberta_score()
    df = roberta_score()  # Assuming this function returns the dataframe
    
    # Get the mean label for each course from the provided test.csv file
    mean_labels = calculate_mean_labels(label_file_path)
    
    # List to hold the final filtered data
    filtered_data = []
    
    # Iterate through each row in the dataframe
    for idx, row in df.iterrows():
        course_id = row['CourseId']
        tokens = row['tokens']
        sentiment_scores = row['sentiment_scores']
        
        # Calculate entropy for each token in the course
        entropies = [shannon_entropy([token]) for token in tokens]
        
        # Filter tokens based on the entropy threshold of 0.7
        filtered_sentiment_scores = [
            score for score, entropy in zip(sentiment_scores, entropies) if entropy <= 0.7
        ]
        
        # Calculate the label as the average of filtered sentiment scores
        label = mean_labels.get(course_id, np.nan)  # Get label from the mean_labels dictionary
        
        # Append the filtered sentiment scores and the label to the filtered_data list
        filtered_data.append({
            'CourseId': course_id,
            'filtered_sentiment_scores': filtered_sentiment_scores,
            'Label': label
        })
    
    # Convert the filtered data into a DataFrame
    filtered_df = pd.DataFrame(filtered_data)
    
    return filtered_df

def final_sentiscore():
    result_df = calculate_filtered_scores('main-.csv')
    return result_df
c= final_sentiscore()
print ("                       entropy                                 ")
print (c)
