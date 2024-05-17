""" Module to provide inference function """
import numpy as np
import torch
import yaml

from src.common.loading import load_pickle, load_trained_model
from src.dataloader.dataloader import load_test_data
from src.common.keywords import (
            MODEL_KEY,
            MODEL_PATH_KEY,
            X_KEY,
            Y_KEY,
            X_TEST_KEY,
            Y_TEST_KEY,
            BATCH_X_KEY,
            TARGET_SCALER_KEY,
            INPUT_SCALER_KEY,
            BATCH_Y_PRED_KEY,
            TEST_LOADER_KEY
            )

def main():
    """
    This function serves as the entry point for your program. It performs the following steps:
    1. Loads configuration settings from YAML files.
    2. Loads the test dataset.
    3. Loads a trained machine learning model.
    4. Retrieves an example batch from the test dataset.
    5. Makes predictions using the loaded model.
    6. Prints relevant information for debugging purposes.

    Returns:
        None
    """
    with open("src/config/config.yaml", "r", encoding="utf8") as file:
        config_common = yaml.safe_load(file)
    with open("src/config/config_predict.yaml", "r", encoding="utf8") as file:
        config_predict = yaml.safe_load(file)
        config = {**config_common, **config_predict}
    # load test dataset
    test_data_dict = load_test_data(config)
    # Load trained model
    load_trained_model_input_dict = {
        MODEL_PATH_KEY: config[MODEL_PATH_KEY],
        X_KEY: test_data_dict[X_TEST_KEY],
        Y_KEY: test_data_dict[Y_TEST_KEY],
    }
    load_trained_model_output_dict = load_trained_model(load_trained_model_input_dict)
    # Load example from test dataset
    batch_x, batch_y = next(iter(test_data_dict[TEST_LOADER_KEY]))
    pkl_dict = load_pickle()
    predict_input_dict = {
        MODEL_KEY: load_trained_model_output_dict[MODEL_KEY],
        BATCH_X_KEY: batch_x,
        TARGET_SCALER_KEY: pkl_dict[TARGET_SCALER_KEY],
        INPUT_SCALER_KEY: pkl_dict[INPUT_SCALER_KEY],
    }
    predict_output_dict = predict(predict_input_dict)
    np.set_printoptions(precision=4, suppress=True)
    print(f"predict_input_dict={predict_input_dict}")
    print(f"predict_output_dict={predict_output_dict[BATCH_Y_PRED_KEY]}")
    batch_y_squeezed = batch_y.reshape((batch_y.shape[0], batch_y.shape[2])).numpy()
    print(f"gt_output_dict={batch_y_squeezed}")


def predict(predict_input_dict):
    """
    Predicts target values using a trained model.

    Args:
        predict_input_dict (dict): A dictionary containing the following keys:
            - BATCH_X_KEY: Input data (features) in batch format.
            - INPUT_SCALER_KEY: Scaler for input data transformation.
            - MODEL_KEY: Trained machine learning model.
            - TARGET_SCALER_KEY: Scaler for target variable transformation.

    Returns:
        dict: A dictionary containing the predicted target values.
            - BATCH_Y_PRED_KEY: Predicted target values (original scale).
    """
    with torch.no_grad():
        batch_x = predict_input_dict[BATCH_X_KEY]
        batch_x = batch_x.reshape((batch_x.shape[0], batch_x.shape[2]))
        input_tensor = torch.from_numpy(
            predict_input_dict[INPUT_SCALER_KEY].transform(batch_x)
        )
        input_tensor = input_tensor.clone().detach().type(torch.float32)
        batch_y_pred = predict_input_dict[MODEL_KEY](input_tensor)
        batch_y_pred_original = predict_input_dict[TARGET_SCALER_KEY].inverse_transform(
            batch_y_pred
        )
        predict_output_dict = {BATCH_Y_PRED_KEY: batch_y_pred_original}
        return predict_output_dict


if __name__ == "__main__":
    main()
