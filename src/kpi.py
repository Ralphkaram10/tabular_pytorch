# predict.py
import torch
import yaml
import torch.nn as nn
import pandas as pd
from src.dataloader.dataloader import CustomDataset  # Assuming dataloader.py is in the correct directory
from torch.utils.data import DataLoader
from src.models.models import SimpleMlp
from src.common.loading import load_test_data, load_trained_model


def main():
    with open("src/config/config_predict.yaml", "r") as f:
        config = yaml.safe_load(f)
    # load test dataset
    test_data_dict =load_test_data(config)
    # Load trained model
    load_trained_model_input_dict  = {'model_path':config['model_path'],'X':test_data_dict['X_test'],'y':test_data_dict['y_test']}
    load_trained_model_output_dict  = load_trained_model(load_trained_model_input_dict)
    # evaluate model
    eval_input_dict = {'test_loader':test_data_dict['test_loader'],'model':load_trained_model_output_dict['model'],'criterion':nn.MSELoss()}
    eval_output_dict = eval(eval_input_dict)
    print(eval_output_dict)




def eval(input_dict):
    input_dict['model'].eval()
    total_val_loss= 0
    with torch.no_grad():
        for batch_X, batch_y in input_dict["test_loader"]:
            pred_y = input_dict['model'](batch_X)
            loss = input_dict['criterion'](pred_y, batch_y)
            total_val_loss += loss.item()
        val_loss = total_val_loss / len(input_dict["test_loader"])
    output_dict={'test_loss':val_loss}
    return output_dict



if __name__ == "__main__":
    main()
