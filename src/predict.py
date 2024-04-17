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
    batch_X,batch_y=next(iter(test_data_dict['test_loader']))
    predict_input_dict={'model':load_trained_model_output_dict['model'],'batch_X':batch_X }
    predict_output_dict=predict(predict_input_dict)
    print(f'predict_input_dict={predict_input_dict}')
    print(f'predict_output_dict={predict_output_dict}')


def predict(predict_input_dict):
    with torch.no_grad():
        input_tensor = torch.tensor(predict_input_dict['batch_X'],dtype=torch.float32)
        batch_y_pred= predict_input_dict['model'](input_tensor)
        predict_output_dict={'batch_y_pred':batch_y_pred}
        return predict_output_dict

if __name__=='__main__':
    main()
