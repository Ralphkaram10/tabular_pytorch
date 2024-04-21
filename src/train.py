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


def main():
    with open("src/config/config_train.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_model(config)


def train_model(config):
    """
    Trains the model
    """
    # load data
    train_val_test_data_dict = load_train_val_test_data(config)
    # Initialize model and optimizer
    init_input_dict = {
        'X': train_val_test_data_dict['X_train'],
        'y': train_val_test_data_dict['y_train']
    }
    init_output_dict = init(init_input_dict)
    # Training loop
    pbar = tqdm(range(config["num_epochs"]))
    train_val_logger=TrainValLogger()
    for epoch in pbar:
        pbar.set_description('training')
        train_one_epoch_input_dict = {
            **init_output_dict, 'train_loader':
            train_val_test_data_dict['train_loader']
        }
        train_one_epoch_output_dict = train_one_epoch(
            train_one_epoch_input_dict)
        eval_one_epoch_input_dict = {
            **train_one_epoch_output_dict, 'val_loader':
            train_val_test_data_dict['val_loader']
        }
        eval_one_epoch_output_dict = eval_one_epoch(eval_one_epoch_input_dict)
        one_epoch_dict={**train_one_epoch_output_dict,**eval_one_epoch_output_dict}
        train_val_logger.append(one_epoch_dict)
        print(
            f"Epoch [{epoch+1}/{config['num_epochs']}] - loss: {train_one_epoch_output_dict['train_loss']:.4f} - val_loss: {eval_one_epoch_output_dict['val_loss']:.4f}"
        )
    train_val_logger.plot_train_val_loss()
    # Save trained model (optional)
    torch.save(init_output_dict['model'].state_dict(), config['model_path'])


def init(init_input_dict):
    """
    Initializes the model and optimizer for training.
    """
    X = init_input_dict["X"]
    y = init_input_dict["y"]
    input_size = X.shape[1]
    output_size = y.shape[1]
    model = SimpleMlp(input_size, output_size)
    init_output_dict = {
        'model': model,
        'optimizer': optim.Adam(model.parameters(), lr=0.001),
        'criterion': nn.MSELoss(),
    }
    return init_output_dict


def load_train_val_test_data(config):
    # Load data using the custom dataset
    train_dataset = CustomDataset({
        'table_path': config["train_table_path"],
        'phase': "train"
    })
    val_dataset = CustomDataset({
        'table_path': config["val_table_path"],
        'phase': "val"
    })
    test_dataset = CustomDataset({
        'table_path': config["test_table_path"],
        'phase': "test"
    })

    X_train = train_dataset.X_df.to_numpy()
    X_val = val_dataset.X_df.to_numpy()
    X_test = test_dataset.X_df.to_numpy()

    y_train = train_dataset.y_df.to_numpy()
    y_val = val_dataset.y_df.to_numpy()
    y_test = test_dataset.y_df.to_numpy()

    train_loader = DataLoader(train_dataset,
                              batch_size=config['batch_size'],
                              shuffle=config['shuffle'])
    val_loader = DataLoader(val_dataset,
                            batch_size=config['batch_size'],
                            shuffle=config['shuffle'])
    test_loader = DataLoader(test_dataset,
                             batch_size=config['batch_size'],
                             shuffle=config['shuffle'])

    output_dict = {
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader
    }
    return output_dict


def train_one_epoch(input_dict):
    input_dict['model'].train()
    total_loss = 0
    for batch_X, batch_y in input_dict["train_loader"]:
        input_dict['optimizer'].zero_grad()
        pred_y = input_dict['model'](batch_X)
        loss = input_dict['criterion'](pred_y, batch_y)
        total_loss += loss.item()
        loss.backward()
        input_dict['optimizer'].step()
    train_loss = total_loss / len(input_dict["train_loader"])
    output_dict = {
        'train_loss': train_loss,
        'model': input_dict['model'],
        'criterion': input_dict['criterion']
    }
    return output_dict


def eval_one_epoch(input_dict):
    input_dict['model'].eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in input_dict["val_loader"]:
            pred_y = input_dict['model'](batch_X)
            loss = input_dict['criterion'](pred_y, batch_y)
            total_val_loss += loss.item()
        val_loss = total_val_loss / len(input_dict["val_loader"])
    output_dict = {'val_loss': val_loss, 'model': input_dict['model']}
    return output_dict

class TrainValLogger(object):
    """Class for logging losses

    """
    def __init__(self):
        super(TrainValLogger, self).__init__()
        self.train_losses=[]
        self.val_losses=[]
    def append(self,logger_input_dict):
        self.logger_input_dict = logger_input_dict
        self.train_losses.append(self.logger_input_dict['train_loss'])
        self.val_losses.append(self.logger_input_dict['val_loss'])

    def plot_train_val_loss(self):
        epoch_count = range(1, len(self.train_losses) + 1)
        plt.plot(epoch_count, self.train_losses, label='train_loss')
        # Plot validation loss
        plt.plot(epoch_count, self.val_losses, label='val_loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('output/train_val_loss.pdf')

if __name__ == "__main__":
    main()
