import pandas as pd
import numpy as np
from entropy import final_sentiscore  # Import your function to load the data

def mean_sentiment_per_course(df):
    """
    Calculates the mean sentiment score and mean label value for each course.
    
    Args:
        df (pd.DataFrame): DataFrame containing 'CourseId', 'filtered_sentiment_scores', and 'label'.
    
    Returns:
        pd.DataFrame: DataFrame with 'CourseId', 'mean_filtered_sentiment_score', and 'mean_label'.
    """
    def compute_mean(sentiment_scores):
        # Flatten the list of sentiment scores and compute the mean
        if len(sentiment_scores) > 0:
            return np.mean(sentiment_scores)
        else:
            return np.nan  # Return NaN if there are no sentiment scores

    # Ensure that 'filtered_sentiment_scores' is not empty for any group
    def safe_concatenate(group):
        # Check if the group is a list of numpy arrays
        if isinstance(group, pd.Series):
            group = group.tolist()  # Convert to a list of arrays
    
        # Flatten the list and compute the mean if valid
        flattened = [item for sublist in group for item in (sublist if isinstance(sublist, (list, np.ndarray)) else [])]
        
        return flattened

    # Group by 'CourseId' and calculate the mean sentiment score for each course
    mean_sentiments = df.groupby('CourseId')['filtered_sentiment_scores'].apply(
        lambda x: compute_mean(safe_concatenate(x))  # Flatten and compute the mean
    ).reset_index()

    # Compute mean label values for each course
    mean_labels = df.groupby('CourseId')['Label'].apply(np.mean).reset_index()

    # Merge the mean sentiment and mean label dataframes on 'CourseId'
    result = pd.merge(mean_sentiments, mean_labels, on='CourseId', how='left')

    # Rename columns for clarity
    result.columns = ['CourseId', 'mean_filtered_sentiment_score', 'mean_label']
    
    return result


def mean_score():
    """
    Fetches the DataFrame from the final_sentiscore function and computes mean sentiment and labels.
    
    Returns:
        pd.DataFrame: DataFrame with 'CourseId', 'mean_filtered_sentiment_score', and 'mean_label'.
    """
    df = final_sentiscore()  # This should return the DataFrame with 'CourseId', 'filtered_sentiment_scores', and 'label'

    mean_sentiment = mean_sentiment_per_course(df)  # Compute the mean sentiment score and mean label per course

    return mean_sentiment


c= mean_score()
print ("                       aggregate                                 ")
print (c)
