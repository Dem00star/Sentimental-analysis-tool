import pandas as pd

def read_data(file_path):
    # Reads the CSV file and returns the data as a DataFrame
    data = pd.read_csv(file_path)
    return data  