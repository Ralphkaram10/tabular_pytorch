import torch
import yaml
import pickle
from src.models.models import SimpleMlp
from src.common.utils import get_abs_path


def load_trained_model(load_trained_model_input_dict):
    """
    Initializes the model
    """
    X = load_trained_model_input_dict["X"]
    y = load_trained_model_input_dict["y"]
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = SimpleMlp(input_size, output_size)
    model_path=get_abs_path(load_trained_model_input_dict["model_path"])
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    load_trained_model_output_dict = {"model": model}
    return load_trained_model_output_dict


def load_pickle():
    config_path=get_abs_path("src/config/config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    output_pickle_path=get_abs_path(config["output_pickle_path"])
    with open(output_pickle_path, "rb") as pkl_file:
        pkl_dict = pickle.load(pkl_file)
    return pkl_dict
