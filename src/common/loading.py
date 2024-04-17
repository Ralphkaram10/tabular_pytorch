import torch
import yaml
import torch.nn as nn
import pandas as pd
from src.dataloader.dataloader import CustomDataset  # Assuming dataloader.py is in the correct directory
from torch.utils.data import DataLoader
from src.models.models import SimpleMlp

def load_test_data(config):
    # Load data using the custom dataset
    test_dataset = CustomDataset(
        input_table_path=config["test_input_table_path"],
        target_table_path=config["test_target_table_path"])

    X_test = test_dataset.X_df.to_numpy()
    y_test = test_dataset.y_df.to_numpy()

    test_loader = DataLoader(test_dataset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle'])


    output_dict = {
        "X_test": X_test,
        "y_test": y_test,
        "test_loader": test_loader
    }
    return output_dict


def load_trained_model(load_trained_model_input_dict):
    """
    Initializes the model
    """
    X = load_trained_model_input_dict["X"]
    y = load_trained_model_input_dict["y"]
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = SimpleMlp(input_size, output_size)
    model.load_state_dict(torch.load(load_trained_model_input_dict['model_path']))
    model.eval()  # Set to evaluation mode
    load_trained_model_output_dict = {'model':model}
    return load_trained_model_output_dict 
