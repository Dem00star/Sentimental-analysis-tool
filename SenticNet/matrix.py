import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load CSV data into pandas DataFrame
df = pd.read_csv("fuzzy_ratings.csv")  # replace with your actual file path

# Define a function to calculate and plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, title):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Create a custom plot with improved aesthetics
    plt.figure(figsize=(12, 6))  # Landscape orientation (wider)
    
    # Using Seaborn's heatmap for better visual appeal
    sns.set(font_scale=1.2)  # Increase font size for better readability
    sns.set_style("whitegrid")  # Set background style
    
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                     xticklabels=[1, 2, 3, 4, 5], yticklabels=[1, 2, 3, 4, 5],
                     cbar=False, annot_kws={"size": 16}, linewidths=1, linecolor='black')
    
    # Add titles and labels with custom fonts
    plt.title(f"{title} - Confusion Matrix", fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Ratings', fontsize=14, fontweight='bold')
    plt.ylabel('True Ratings', fontsize=14, fontweight='bold')
    
    # Rotate axis labels for better clarity
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    # Show the plot
    plt.tight_layout()
    plt.show()

# Define fuzzy membership types
membership_types = ['Gaussian', 'Sigmoidal', 'Triangular', 'Trapezoidal', 'Piecewise']

# Iterate through each membership type and generate the confusion matrix
for mtype in membership_types:
    predicted_column = f'Rating_{mtype}'
    
    # Round the predicted ratings to the nearest integer
    predicted_labels = df[predicted_column].round().astype(int)
    
    # Get the true labels from the 'mean_label' column
    true_labels = df['mean_label'].astype(int)
    
    # Plot confusion matrix with custom styling
    plot_confusion_matrix(true_labels, predicted_labels, f"Confusion Matrix for {mtype}")