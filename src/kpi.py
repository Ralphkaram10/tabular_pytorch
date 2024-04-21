# predict.py
import torch
import yaml
import torch.nn as nn
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from src.dataloader.dataloader import CustomDataset, load_test_data  # Assuming dataloader.py is in the correct directory
from torch.utils.data import DataLoader
from src.models.models import SimpleMlp
from src.common.loading import load_trained_model, load_pickle


def main():
    with open("src/config/config_predict.yaml", "r") as f:
        config = yaml.safe_load(f)
    # load test dataset
    test_data_dict = load_test_data(config)
    # Load trained model
    load_trained_model_input_dict = {
        'model_path': config['model_path'],
        'X': test_data_dict['X_test'],
        'y': test_data_dict['y_test']
    }
    load_trained_model_output_dict = load_trained_model(
        load_trained_model_input_dict)
    # evaluate model
    pkl_dict = load_pickle()
    criterions_dict= {'mse':mean_squared_error, 'r2':r2_score}
    eval_input_dict = {
        'test_loader': test_data_dict['test_loader'],
        'model': load_trained_model_output_dict['model'],
        'input_scaler': pkl_dict['input_scaler'],
        'target_scaler': pkl_dict['target_scaler']
    }
    eval_output_dict = eval(eval_input_dict)
    print(eval_output_dict)

def compute_metrics_per_batch(input_dict):
    mse=mean_squared_error(input_dict['batch_y'],input_dict['batch_y_pred'])
    r2=r2_score(input_dict['batch_y'],input_dict['batch_y_pred'])
    return {'mse':mse, 'r2':r2}


def eval(input_dict):
    input_dict['model'].eval()
    total_test_loss = 0
    total_metrics={'mse':0, 'r2':0}
    with torch.no_grad():
        for batch_X, batch_y in input_dict["test_loader"]:
            batch_X = batch_X.reshape((batch_X.shape[0],batch_X.shape[2]))
            batch_y = batch_y.reshape((batch_y.shape[0],batch_y.shape[2]))
            input_tensor = input_dict['input_scaler'].transform(
                batch_X)
            input_tensor = torch.tensor(input_tensor,dtype=torch.float32)
            pred_y = input_dict['model'](input_tensor)
            pred_y_original = input_dict[
                'target_scaler'].inverse_transform(pred_y)
            metrics_input_dict={'batch_y_pred':pred_y, 'batch_y':batch_y}
            metrics_dict= compute_metrics_per_batch(metrics_input_dict)
            for k in metrics_dict.keys():
                total_metrics[k]+=metrics_dict[k]
        for k in metrics_dict.keys():
            total_metrics[k] = total_metrics[k] / len(input_dict["test_loader"])
    return total_metrics


if __name__ == "__main__":
    main()
