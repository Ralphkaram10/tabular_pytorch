import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.dataloader.dataloader import CustomDataset
from src.models.models import SimpleMlp
from src.common.keywords import (
            BATCH_SIZE_KEY,
            BATCH_Y_KEY,
            MODEL_KEY,
            MODEL_PATH_KEY,
            TEST_KEY,
            TRAIN_KEY,
            X_KEY,
            Y_KEY,
            X_TEST_KEY,
            Y_TEST_KEY,
            X_TRAIN_KEY,
            Y_TRAIN_KEY,
            X_VAL_KEY,
            Y_VAL_KEY,
            BATCH_X_KEY,
            TARGET_SCALER_KEY,
            INPUT_SCALER_KEY,
            BATCH_Y_PRED_KEY,
            TEST_LOADER_KEY,
            TRAIN_LOADER_KEY,
            VAL_LOADER_KEY,
            MSE_KEY,
            RMSE_KEY,
            R2_KEY,
            PLOT_TITLE_KEY,
            TRAIN_DATASET_KEY,
            VAL_DATASET_KEY,
            TEST_DATASET_KEY,
            OPTIMIZER_KEY,
            CRITERION_KEY,
            TRAIN_LOSS_KEY,
            VAL_LOSS_KEY,
            TABLE_PATH_KEY,
            TRAIN_TABLE_PATH_KEY,
            VAL_TABLE_PATH_KEY,
            TEST_TABLE_PATH_KEY,
            PHASE_KEY,
            TRAIN_KEY,
            VAL_KEY,
            TEST_KEY,
            NUM_EPOCHS_KEY,
            SHUFFLE_KEY
            )




def main():
    """
    Entry point for training the model.

    Reads configuration from 'config_train.yaml' and trains the model accordingly.
    """
    with open("src/config/config_train.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_model(config)


def train_model(config):
    """
    Trains a machine learning model using the specified configuration.

    Args:
        config (dict): A dictionary containing configuration parameters.

    Returns:
        None
    """
    # load data
    train_val_test_data_dict = load_train_val_test_data(config)
    # Initialize model and optimizer
    init_input_dict = {
        X_KEY: train_val_test_data_dict[X_TRAIN_KEY],
        Y_KEY: train_val_test_data_dict[Y_TRAIN_KEY],
    }
    init_output_dict = init(init_input_dict)
    # Training loop
    pbar = tqdm(range(config[NUM_EPOCHS_KEY]))
    train_val_logger = TrainValLogger()
    for epoch in pbar:
        pbar.set_description("training")
        train_one_epoch_input_dict = {
            **init_output_dict,
            TRAIN_LOADER_KEY: train_val_test_data_dict[TRAIN_LOADER_KEY],
        }
        train_one_epoch_output_dict = train_one_epoch(train_one_epoch_input_dict)
        eval_one_epoch_input_dict = {
            **train_one_epoch_output_dict,
            VAL_LOADER_KEY: train_val_test_data_dict[VAL_LOADER_KEY],
        }
        eval_one_epoch_output_dict = eval_one_epoch(eval_one_epoch_input_dict)
        one_epoch_dict = {**train_one_epoch_output_dict, **eval_one_epoch_output_dict}
        train_val_logger.append(one_epoch_dict)
        print(
            f"Epoch [{epoch+1}/{config[NUM_EPOCHS_KEY]}] - loss: {train_one_epoch_output_dict[TRAIN_LOSS_KEY]:.4f} - val_loss: {eval_one_epoch_output_dict[VAL_LOSS_KEY]:.4f}"
        )
    train_val_logger.plot_train_val_loss()
    # Save trained model (optional)
    torch.save(init_output_dict[MODEL_KEY].state_dict(), config[MODEL_PATH_KEY])


def init(init_input_dict):
    """
    Initializes a neural network model and optimizer for training.

    Args:
        init_input_dict (dict): A dictionary containing input data (X) and target labels (y).

    Returns:
        dict: A dictionary containing the initialized model, optimizer, and loss criterion.
    """
    X = init_input_dict[X_KEY]
    y = init_input_dict[Y_KEY]
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = SimpleMlp(input_size, output_size)
    init_output_dict = {
        MODEL_KEY: model,
        OPTIMIZER_KEY: optim.Adam(model.parameters(), lr=0.001),
        CRITERION_KEY: nn.MSELoss(),
    }
    return init_output_dict


def load_train_val_test_data(config):
    """
    Load and preprocess train, validation, and test data using a custom dataset.

    Args:
        config (dict): A dictionary containing configuration parameters.
            - TRAIN_TABLE_PATH_KEY: Path to the training data table.
            - VAL_TABLE_PATH_KEY: Path to the validation data table.
            - TEST_TABLE_PATH_KEY: Path to the test data table.
            - BATCH_SIZE_KEY: Batch size for data loaders.
            - SHUFFLE_KEY: Whether to shuffle the data during loading.

    Returns:
        dict: A dictionary containing the following keys:
            - 'train_dataset': CustomDataset for training data.
            - 'val_dataset': CustomDataset for validation data.
            - 'test_dataset': CustomDataset for test data.
            - 'X_train': Numpy array of training features.
            - 'X_val': Numpy array of validation features.
            - 'X_test': Numpy array of test features.
            - 'y_train': Numpy array of training labels.
            - 'y_val': Numpy array of validation labels.
            - 'y_test': Numpy array of test labels.
            - 'train_loader': DataLoader for training data.
            - 'val_loader': DataLoader for validation data.
            - 'test_loader': DataLoader for test data.
    """
    # Load data using the custom dataset
    train_dataset = CustomDataset(
        {TABLE_PATH_KEY: config[TRAIN_TABLE_PATH_KEY], PHASE_KEY: TRAIN_KEY}
    )
    val_dataset = CustomDataset(
        {TABLE_PATH_KEY: config[VAL_TABLE_PATH_KEY], PHASE_KEY: VAL_KEY}
    )
    test_dataset = CustomDataset(
        {TABLE_PATH_KEY: config[TEST_TABLE_PATH_KEY], PHASE_KEY: TEST_KEY}
    )

    X_train = train_dataset.X_df.to_numpy()
    X_val = val_dataset.X_df.to_numpy()
    X_test = test_dataset.X_df.to_numpy()

    y_train = train_dataset.y_df.to_numpy()
    y_val = val_dataset.y_df.to_numpy()
    y_test = test_dataset.y_df.to_numpy()

    train_loader = DataLoader(
        train_dataset, batch_size=config[BATCH_SIZE_KEY], shuffle=config[SHUFFLE_KEY]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config[BATCH_SIZE_KEY], shuffle=config[SHUFFLE_KEY]
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config[BATCH_SIZE_KEY], shuffle=config[SHUFFLE_KEY]
    )

    output_dict = {
        TRAIN_DATASET_KEY: train_dataset,
        VAL_DATASET_KEY: val_dataset,
        TEST_DATASET_KEY: test_dataset,
        X_TRAIN_KEY: X_train,
        X_VAL_KEY: X_val,
        X_TEST_KEY: X_test,
        Y_TRAIN_KEY: y_train,
        Y_VAL_KEY: y_val,
        Y_TEST_KEY: y_test,
        TRAIN_LOADER_KEY: train_loader,
        VAL_LOADER_KEY: val_loader,
        TEST_LOADER_KEY: test_loader,
    }
    return output_dict


def train_one_epoch(input_dict):
    """
    Trains a machine learning model for one epoch.

    Args:
        input_dict (dict): A dictionary containing the following keys:
            - MODEL_KEY: The machine learning model to be trained.
            - TRAIN_LOADER_KEY: The data loader for training data.
            - OPTIMIZER_KEY: The optimizer used for gradient updates.
            - CRITERION_KEY: The loss function for training.

    Returns:
        dict: A dictionary containing the following keys:
            - TRAIN_LOSS_KEY: Average training loss for the epoch.
            - MODEL_KEY: Updated machine learning model.
            - CRITERION_KEY: Loss function used for training.
    """
    input_dict[MODEL_KEY].train()
    total_loss = 0
    for batch_X, batch_y in input_dict[TRAIN_LOADER_KEY]:
        input_dict[OPTIMIZER_KEY].zero_grad()
        pred_y = input_dict[MODEL_KEY](batch_X)
        loss = input_dict[CRITERION_KEY](pred_y, batch_y)
        total_loss += loss.item()
        loss.backward()
        input_dict[OPTIMIZER_KEY].step()
    train_loss = total_loss / len(input_dict[TRAIN_LOADER_KEY])
    output_dict = {
        TRAIN_LOSS_KEY: train_loss,
        MODEL_KEY: input_dict[MODEL_KEY],
        CRITERION_KEY: input_dict[CRITERION_KEY],
    }
    return output_dict


def eval_one_epoch(input_dict):
    """
    Evaluate the model on the validation dataset for one epoch.

    Args:
        input_dict (dict): A dictionary containing the following keys:
            - MODEL_KEY: The PyTorch model to evaluate.
            - VAL_LOADER_KEY: The validation data loader.
            - CRITERION_KEY: The loss criterion.

    Returns:
        dict: A dictionary containing the validation loss (VAL_LOSS_KEY) and the model (MODEL_KEY).
    """
    input_dict[MODEL_KEY].eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in input_dict[VAL_LOADER_KEY]:
            pred_y = input_dict[MODEL_KEY](batch_X)
            loss = input_dict[CRITERION_KEY](pred_y, batch_y)
            total_val_loss += loss.item()
        val_loss = total_val_loss / len(input_dict[VAL_LOADER_KEY])
    output_dict = {VAL_LOSS_KEY: val_loss, MODEL_KEY: input_dict[MODEL_KEY]}
    return output_dict


class TrainValLogger(object):
    """
    Class for logging training and validation losses during model training.

    Attributes:
        train_losses (list): List to store training losses.
        val_losses (list): List to store validation losses.

    Methods:
        append(logger_input_dict):
            Appends training and validation losses to the respective lists.
            Args:
                logger_input_dict (dict): A dictionary containing training and validation loss values.
                    Expected keys: 'train_loss' and 'val_loss'.
        plot_train_val_loss():
            Plots the training and validation losses over epochs.
    """
    def __init__(self):
        super(TrainValLogger, self).__init__()
        self.train_losses = []
        self.val_losses = []

    def append(self, logger_input_dict):
        """
        Appends training and validation losses to the respective lists.

        Args:
            logger_input_dict (dict): A dictionary containing training and validation loss values.
                Expected keys: 'train_loss' and 'val_loss'.
        """
        self.logger_input_dict = logger_input_dict
        self.train_losses.append(self.logger_input_dict[TRAIN_LOSS_KEY])
        self.val_losses.append(self.logger_input_dict[VAL_LOSS_KEY])

    def plot_train_val_loss(self):
        """
        Plots the training and validation losses over epochs.
        """
        epoch_count = range(1, len(self.train_losses) + 1)
        plt.plot(epoch_count, self.train_losses, label=TRAIN_LOSS_KEY)
        # Plot validation loss
        plt.plot(epoch_count, self.val_losses, label=VAL_LOSS_KEY)
        plt.title("Model Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("output/train_val_loss.pdf")


if __name__ == "__main__":
    main()
