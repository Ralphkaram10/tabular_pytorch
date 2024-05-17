""" Module to compute key performance indicators """
import torch
import yaml
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from matplotlib import pyplot as plt

# Assuming dataloader.py is in the correct directory
from src.dataloader.dataloader import load_test_data, load_test_data_batch_size_1
from src.common.loading import load_trained_model, load_pickle
from src.predict import predict
from src.common.keywords import (
            BATCH_Y_KEY,
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
            TEST_LOADER_KEY,
            MSE_KEY,
            RMSE_KEY,
            R2_KEY,
            PLOT_TITLE_KEY,
            )



def main():
    """
    This function performs the following steps:
    1. Loads the test dataset from the specified configuration file.
    2. Loads a trained machine learning model using the provided model path.
    3. Evaluates the model's performance on the test data.
    4. Plots the ground truth vs. prediction for visualization.

    Args:
        None (No explicit arguments are passed to this function.)

    Returns:
        None (The function performs the specified tasks without returning any value.)
    """
    with open("src/config/config_predict.yaml", "r") as f:
        config = yaml.safe_load(f)
    # load test dataset
    test_data_dict = load_test_data(config)
    # Load trained model
    load_trained_model_input_dict = {
        MODEL_PATH_KEY: config[MODEL_PATH_KEY],
        X_KEY: test_data_dict[X_TEST_KEY],
        Y_KEY: test_data_dict[Y_TEST_KEY],
    }
    load_trained_model_output_dict = load_trained_model(
        load_trained_model_input_dict)
    # evaluate model
    pkl_dict = load_pickle()
    evaluate_input_dict = {
        TEST_LOADER_KEY: test_data_dict[TEST_LOADER_KEY],
        MODEL_KEY: load_trained_model_output_dict[MODEL_KEY],
        INPUT_SCALER_KEY: pkl_dict[INPUT_SCALER_KEY],
        TARGET_SCALER_KEY: pkl_dict[TARGET_SCALER_KEY],
    }
    evaluate_output_dict = evaluate(evaluate_input_dict)
    print(evaluate_output_dict)
    # plot ground truth vs prediction
    plot_ground_truth_vs_prediction({**evaluate_input_dict, **config})


def compute_metrics_per_batch(input_dict):
    """
    Computes various metrics for a batch of predictions.

    Args:
        input_dict (dict): A dictionary containing the following keys:
            - BATCH_Y_KEY: The ground truth labels for the batch.
            - BATCH_Y_PRED_KEY: The predicted labels for the batch.

    Returns:
        dict: A dictionary containing the computed metrics:
            - 'mse': Mean Squared Error
            - 'rmse': Root Mean Squared Error
            - 'r2': R-squared (coefficient of determination)
    """
    mse = mean_squared_error(input_dict[BATCH_Y_KEY], input_dict[BATCH_Y_PRED_KEY])
    rmse = root_mean_squared_error(
        input_dict[BATCH_Y_KEY], input_dict[BATCH_Y_PRED_KEY])
    r2 = r2_score(input_dict[BATCH_Y_KEY], input_dict[BATCH_Y_PRED_KEY])
    return {MSE_KEY: mse, RMSE_KEY: rmse, R2_KEY: r2}


def plot_ground_truth_vs_prediction(input_dict):
    """
    Plots a scatter plot comparing ground truth values (y_real) with predicted values (y_pred).

    Args:
        input_dict (dict): A dictionary containing necessary input data.

    Returns:
        None: The plot is saved as a PDF file.

    Example usage:
        input_dict = {
            X_TEST_KEY: ...,
            Y_TEST_KEY: ...,
            BATCH_Y_PRED_KEY: ...,
            PLOT_TITLE_KEY: 'Ground Truth vs. Prediction'
        }
        plot_ground_truth_vs_prediction(input_dict)
    """
    test_data_dict = load_test_data_batch_size_1(input_dict)
    y_test = test_data_dict[Y_TEST_KEY].squeeze()
    predict_input_dict = {
        **input_dict, BATCH_X_KEY: np.expand_dims(test_data_dict[X_TEST_KEY], axis=1)}
    predict_output_dict = predict(predict_input_dict)
    y_pred = predict_output_dict[BATCH_Y_PRED_KEY].squeeze()
    plt.scatter(y_test, y_pred, color="blue")
    plot_title = input_dict[PLOT_TITLE_KEY]
    plt.title(plot_title)
    plt.xlabel("y_real")
    plt.ylabel("y_pred")
    plt.savefig(f"output/{plot_title}.pdf")


def evaluate(input_dict):
    """
    Evaluates the performance metrics of a machine learning model on a test dataset.

    Args:
        input_dict (dict): A dictionary containing relevant inputs, including:
            - MODEL_KEY: The trained machine learning model.
            - TEST_LOADER_KEY: The data loader for the test dataset.
            - BATCH_X_KEY: The input features for each batch.
            - BATCH_Y_KEY: The ground truth labels for each batch.

    Returns:
        dict: A dictionary containing the computed metrics, including:
            - MSE_KEY: Mean Squared Error.
            - RMSE_KEY: Root Mean Squared Error.
            - R2_KEY: R-squared (coefficient of determination).
    """
    input_dict[MODEL_KEY].eval()
    total_metrics = {MSE_KEY: 0, RMSE_KEY: 0, R2_KEY: 0}
    with torch.no_grad():
        metrics_dict = {}
        for batch_x, batch_y in input_dict[TEST_LOADER_KEY]:
            batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[2]))
            predict_input_dict = {**input_dict, BATCH_X_KEY: batch_x}
            predict_output_dict = predict(predict_input_dict)
            metrics_input_dict = {
                BATCH_Y_PRED_KEY: predict_output_dict[BATCH_Y_PRED_KEY], BATCH_Y_KEY: batch_y}
            metrics_dict = compute_metrics_per_batch(metrics_input_dict)
            for k in metrics_dict:
                total_metrics[k] += metrics_dict[k]
        for k in metrics_dict:
            total_metrics[k] = total_metrics[k] / \
                len(input_dict[TEST_LOADER_KEY])
    return total_metrics


if __name__ == "__main__":
    main()
