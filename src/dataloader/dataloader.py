"""Module providing pytorch Dataset for regression/classification in tabular datasets."""
import os
import yaml
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from src.common.utils import get_abs_path

class CustomDataset(Dataset):
    """Class representing pytorch Dataset for table regression/classification"""

    def __init__(self, custom_dataset_input_dict):
        self.custom_dataset_input_dict = custom_dataset_input_dict
        config_path=get_abs_path("src/config/config.yaml")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        df = pd.read_csv(get_abs_path(custom_dataset_input_dict["table_path"]), sep=";")
        self.X_df = df[self.config["input_columns"]]
        self.y_df = df[self.config["output_columns"]]
        self.X = self.X_df.to_numpy(dtype=np.float32)
        self.y = self.y_df.to_numpy(dtype=np.float32)
        if custom_dataset_input_dict["phase"] == "train":
            self.dump_pickle()
        if (
            custom_dataset_input_dict["phase"] == "val"
            or custom_dataset_input_dict["phase"] == "test"
        ):
            self.load_pickle()

    def dump_pickle(self):
        self.input_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        # Fit the scalers to the data
        self.input_scaler.fit(self.X)
        self.target_scaler.fit(self.y)

        pkl_dict = {
            "input_scaler": self.input_scaler,
            "target_scaler": self.target_scaler,
        }
        output_pickle_path=get_abs_path(self.config["output_pickle_path"])
        with open(output_pickle_path, "wb") as pkl_file:
            pickle.dump(pkl_dict, pkl_file)

    def load_pickle(self):
        output_pickle_path=get_abs_path(self.config["output_pickle_path"])
        with open(output_pickle_path, "rb") as pkl_file:
            pkl_dict = pickle.load(pkl_file)
            self.input_scaler = pkl_dict["input_scaler"]
            self.target_scaler = pkl_dict["target_scaler"]

    def __len__(self):
        return len(self.X_df)

    def __getitem__(self, idx):
        # Load input
        X = self.X[idx].reshape(1, -1)
        y = self.y[idx].reshape(1, -1)
        if (
            self.custom_dataset_input_dict.get("phase") == "train"
            or self.custom_dataset_input_dict.get("phase") == "val"
        ):
            X_scaled = self.input_scaler.transform(X)
            y_scaled = self.target_scaler.transform(y)
            return X_scaled.astype(np.float32), y_scaled.astype(np.float32)
        elif self.custom_dataset_input_dict.get("phase") == "test":
            return X.astype(np.float32), y.astype(np.float32)


def load_test_data(config):
    # Load data using the custom dataset
    test_dataset = CustomDataset(
        {"table_path": config["test_table_path"], "phase": "test"}
    )

    X_test = test_dataset.X_df.to_numpy()
    y_test = test_dataset.y_df.to_numpy()

    test_loader = DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=config["shuffle"]
    )

    output_dict = {"X_test": X_test, "y_test": y_test, "test_loader": test_loader}
    return output_dict

def load_test_data_batch_size_1(config):
    # Load data using the custom dataset
    test_dataset = CustomDataset(
        {"table_path": config["test_table_path"], "phase": "test"}
    )

    X_test = test_dataset.X_df.to_numpy()
    y_test = test_dataset.y_df.to_numpy()

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=config["shuffle"]
    )

    output_dict = {"X_test": X_test, "y_test": y_test, "test_loader": test_loader}
    return output_dict



if __name__ == "__main__":
    custom_dataset_input_dict = {
        "table_path": "data/auction+verification/data.csv",
        "phase": "predict",
    }
    dataset = CustomDataset(custom_dataset_input_dict)
