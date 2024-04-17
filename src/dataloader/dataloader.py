"""Module providing pytorch Dataset for regression/classification in tabular datasets."""
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

class CustomDataset(Dataset):
    """Class representing pytorch Dataset for table regression/classification"""

    def __init__(self, input_table_path="", target_table_path=""):
        self.X_df = pd.read_csv(input_table_path, sep=";")
        self.y_df = pd.read_csv(target_table_path, sep=";")
        self.X = self.X_df.to_numpy(dtype=np.float32)
        self.y = self.y_df.to_numpy(dtype=np.float32)
        self.input_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        # Fit the scalers to the data
        self.input_scaler.fit(self.X)
        self.target_scaler.fit(self.y)

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        # Load input
        X = self.X[idx].reshape(1,-1)        
        y = self.y[idx].reshape(1,-1) 
        X_scaled = self.input_scaler.transform(X)
        y_scaled = self.target_scaler.transform(y)
        return X_scaled.astype(np.float32), y_scaled.astype(np.float32)

if __name__=='__main__':
    input_table_path="data/auction+verification/data.csv"
    target_table_path="data/auction+verification/data.csv"
    dataset=CustomDataset(input_table_path=input_table_path,target_table_path=target_table_path)
