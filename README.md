
#  Tabular PyTorch

This repository is used to train and evaluate deep learning models for tabular data regression.  

>**How to Use**:
>  * Install the necessary python dependencies (found in requirements.txt) preferably in a venv or conda environment.
>  * The needed configuration files needed to for this repository are located at src/config. Update them when needed.
>  * Use train.py to train the model 
>  * Use kpi.py to evaluate the model
>  * Detailed instructions on how to use this repo are found in the following sections


## Machine Learning Model Training (train.py):

* Purpose: Focuses on training a machine learning model using PyTorch.  
Components:
  * Initialization: Initialize the neural network model, optimizer, and loss criterion.
  * Data Loading and Preprocessing: Load and preprocess training, validation, and test data using a custom dataset.
  * Training Loop: Train the model, updating weights based on gradients.
  * Validation Loop: Evaluate the model on the validation dataset for each epoch.
  * Logging: Provides a class for logging training and validation losses during model training.

>**How to Use**:
>  * When needed, update the configuration file of "src/train.py" found at: "src/config/config_train.yaml"
>  and the general configuration file of the repository found at: "src/config/config.yaml".
>  * From the root of the repository execute the script by running `python src/train.py` 



## Inference Function (predict.py):

* Purpose: Provides an inference function for making predictions using a trained machine learning model.
* Steps:
  * Load configuration settings from YAML files.
  * Load the test dataset.
  * Load a pre-trained machine learning model.
  * Retrieve an example batch from the test dataset.
  * Make predictions using the loaded model.
  * Print relevant debugging information.

>**How to Use**:
>  * When needed, update the configuration file of "src/predict.py" found at: "src/config/config_predict.yaml"
>  and the general configuration file of the repository found at: "src/config/config.yaml".
>  * From the root of the repository execute the script by running `python src/predict.py`.


## Module for Computing Key Performance Indicators (kpi.py):

* Purpose: Focuses on evaluating a machine learning (tabular data regression) model.
* Steps:
  * Load the test dataset from the specified configuration file.
  * Load a trained machine learning model using the provided model path.
  * Evaluate the model’s performance on the test data.
  * Plot the ground truth vs. prediction for visualization.
* Metrics Computed:
  * Mean Squared Error (MSE)
  * Root Mean Squared Error (RMSE)
  * R-squared (coefficient of determination)

>**How to Use**:
>  * When needed, update the configuration file of "src/kpi.py" found at: "src/config/config_predict.yaml"
>  and the general configuration file of the repository found at: "src/config/config.yaml".
>  * From the root of the repository execute the script by running `python src/kpi.py` 


