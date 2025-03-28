# Import libraries
import json
import pandas as pd

# Data directories
json_file_supervised = r"C:/Users/shaya/OneDrive/Desktop\Data\API/supervised_call_graphs.json"
json_file_all = r"C:/Users/shaya/OneDrive/Desktop/Data/API/remaining_call_graphs.json"
bahvior_supervised = r"C:/Users/shaya/OneDrive/Desktop/Data\API/supervised_dataset.csv"
behavior_test = r"C:/Users/shaya/OneDrive/Desktop/Data/API/remaining_behavior_ext.csv"

def json_file_reader(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    #return data
    return pd.DataFrame(data)
    
def csv_file_reader(file_path):
    df_behavior = pd.read_csv(file_path)
    return df_behavior
    
def combiner(df1, df2):
    df_combined = pd.merge(df1, df2, on='_id')
    df_combined = df_combined.drop_duplicates(subset='_id')
    
    return df_combined