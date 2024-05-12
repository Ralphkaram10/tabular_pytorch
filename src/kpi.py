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


def main():
    with open("src/config/config_predict.yaml", "r") as f:
        config = yaml.safe_load(f)
    # load test dataset
    test_data_dict = load_test_data(config)
    # Load trained model
    load_trained_model_input_dict = {
        "model_path": config["model_path"],
        "X": test_data_dict["X_test"],
        "y": test_data_dict["y_test"],
    }
    load_trained_model_output_dict = load_trained_model(load_trained_model_input_dict)
    # evaluate model
    pkl_dict = load_pickle()
    eval_input_dict = {
        "test_loader": test_data_dict["test_loader"],
        "model": load_trained_model_output_dict["model"],
        "input_scaler": pkl_dict["input_scaler"],
        "target_scaler": pkl_dict["target_scaler"],
    }
    eval_output_dict = eval(eval_input_dict)
    print(eval_output_dict)
    # plot ground truth vs prediction
    plot_ground_truth_vs_prediction({**eval_input_dict,**config})


def compute_metrics_per_batch(input_dict):
    mse = mean_squared_error(input_dict["batch_y"], input_dict["batch_y_pred"])
    rmse = root_mean_squared_error(input_dict["batch_y"], input_dict["batch_y_pred"])
    r2 = r2_score(input_dict["batch_y"], input_dict["batch_y_pred"])
    return {"mse": mse, "rmse": rmse, "r2": r2}


def plot_ground_truth_vs_prediction(input_dict):
    test_data_dict=load_test_data_batch_size_1(input_dict)
    y_test=test_data_dict['y_test'].squeeze()
    predict_input_dict={**input_dict,'batch_X':np.expand_dims(test_data_dict['X_test'], axis=1)}
    predict_output_dict=predict(predict_input_dict)
    y_pred=predict_output_dict['batch_y_pred'].squeeze()
    plt.scatter(y_test, y_pred, color="blue")
    plot_title=input_dict['plot_title']
    plt.title(plot_title)
    plt.xlabel("y_real")
    plt.ylabel("y_pred")
    plt.savefig(f"output/{plot_title}.pdf")


def eval(input_dict):
    input_dict["model"].eval()
    total_metrics = {"mse": 0.0, "rmse": 0.0, "r2": 0.0}
    with torch.no_grad():
        metrics_dict={}
        for batch_X, batch_y in input_dict["test_loader"]:
            batch_y = batch_y.reshape((batch_y.shape[0], batch_y.shape[2]))
            predict_input_dict={**input_dict,'batch_X':batch_X}
            predict_output_dict=predict(predict_input_dict)
            metrics_input_dict = {"batch_y_pred": predict_output_dict["batch_y_pred"], "batch_y": batch_y}
            metrics_dict = compute_metrics_per_batch(metrics_input_dict)
            for k in metrics_dict:
                total_metrics[k] += metrics_dict[k]
        for k in metrics_dict:
            total_metrics[k] = total_metrics[k] / len(input_dict["test_loader"])
    return total_metrics


if __name__ == "__main__":
    main()
