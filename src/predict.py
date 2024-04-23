import torch
import yaml
import torch.nn as nn
import pandas as pd
from src.dataloader.dataloader import CustomDataset, load_test_data   # Assuming dataloader.py is in the correct directory
from torch.utils.data import DataLoader
from src.models.models import SimpleMlp
from src.common.loading import load_trained_model, load_pickle


def main():
    with open("src/config/config.yaml", "r") as f:
        config_common= yaml.safe_load(f)
    with open("src/config/config_predict.yaml", "r") as f:
        config_predict= yaml.safe_load(f)
        config={**config_common, **config_predict}
    # load test dataset
    test_data_dict =load_test_data(config)
    # Load trained model
    load_trained_model_input_dict  = {'model_path':config['model_path'],'X':test_data_dict['X_test'],'y':test_data_dict['y_test']}
    load_trained_model_output_dict  = load_trained_model(load_trained_model_input_dict)
    # Load example from test dataset
    batch_X,batch_y=next(iter(test_data_dict['test_loader']))
    pkl_dict= load_pickle()
    predict_input_dict={'model':load_trained_model_output_dict['model'],'batch_X':batch_X, 'target_scaler':pkl_dict['target_scaler'], 'input_scaler':pkl_dict['input_scaler']}
    predict_output_dict=predict(predict_input_dict)
    print(f'predict_input_dict={predict_input_dict}')
    print(f'predict_output_dict={predict_output_dict}')
    batch_y_squeezed= batch_y.reshape((batch_y.shape[0],batch_y.shape[2]))
    print(f'gt_output_dict={batch_y_squeezed}')


def predict(predict_input_dict):
    with torch.no_grad():
        batch_X = torch.tensor(predict_input_dict['batch_X'],dtype=torch.float32)
        batch_X = batch_X.reshape((batch_X.shape[0],batch_X.shape[2]))
        input_tensor = torch.from_numpy(predict_input_dict['input_scaler'].transform(batch_X))
        input_tensor = torch.tensor(input_tensor,dtype=torch.float32)
        batch_y_pred = predict_input_dict['model'](input_tensor)
        batch_y_pred_original = predict_input_dict['target_scaler'].inverse_transform(batch_y_pred)
        predict_output_dict={'batch_y_pred':batch_y_pred_original}
        return predict_output_dict

if __name__=='__main__':
    main()
