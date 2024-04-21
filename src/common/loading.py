import torch
import yaml
import pickle
import torch.nn as nn
import pandas as pd
from src.dataloader.dataloader import CustomDataset  # Assuming dataloader.py is in the correct directory
from torch.utils.data import DataLoader
from src.models.models import SimpleMlp


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

def load_pickle():
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    with open(config["output_pickle_path"], "rb") as pkl_file:
        pkl_dict = pickle.load(pkl_file)
    return pkl_dict
