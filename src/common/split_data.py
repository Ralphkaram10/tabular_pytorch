import pandas as pd
from sklearn.model_selection import train_test_split
import os


def split_dataframe(df,
                    train_percent,
                    val_percent,
                    test_percent,
                    random_state=None):
    """
    Splits a pandas DataFrame into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        train_percent (float): Percentage for the train set.
        val_percent (float): Percentage for the validation set.
        test_percent (float): Percentage for the test set.
        random_state (int, optional): Random seed for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Train, validation, and test DataFrames.
    """
    # Split into train and remaining data
    remaining_data, test_data = train_test_split(df,
                                                 test_size=test_percent/100,
                                                 random_state=random_state)

    # Calculate the adjusted validation percentage based on the remaining data
    adjusted_val_percent = 100*((val_percent/100) / (1 - (test_percent/100)))

    # Split the remaining data into train and validation
    train_data, val_data = train_test_split(remaining_data,
                                            test_size=adjusted_val_percent/100,
                                            random_state=random_state)

    return train_data, val_data, test_data


if __name__ == "__main__":
    tabular_data_path = 'data/auction+verification/data.csv'
    output_dir_path = 'data/auction+verification/'
    train_percent = 80
    val_percent = 10
    test_percent = 10
    random_state = 0

    df = pd.read_csv(tabular_data_path, sep=';')
    train_df, val_df, test_df = split_dataframe(df,
                                                train_percent,
                                                val_percent,
                                                test_percent,
                                                random_state=random_state)

    filename_with_extension = os.path.basename(tabular_data_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    train_output_filename_with_extension = f'train_{filename_without_extension}.csv'
    val_output_filename_with_extension = f'val_{filename_without_extension}.csv'
    test_output_filename_with_extension = f'test_{filename_without_extension}.csv'
    train_output_path = os.path.join(output_dir_path,
                                     train_output_filename_with_extension)
    val_output_path = os.path.join(output_dir_path,
                                   val_output_filename_with_extension)
    test_output_path = os.path.join(output_dir_path,
                                    test_output_filename_with_extension)

    train_df.to_csv(train_output_path, sep=';', index=False)
    val_df.to_csv(val_output_path, sep=';', index=False)
    test_df.to_csv(test_output_path, sep=';', index=False)

    print("Data split and saved successfully")
