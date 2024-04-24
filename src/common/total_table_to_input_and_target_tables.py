import pandas as pd
import os


def transform_total_table_to_input_and_target_tables(
    total_table_path="data/auction+verification/data.csv",
    input_columns=None,
    output_columns=None,
    X_filename="inputs.csv",
    y_filename="targets.csv",
    X_dir_path="data/auction+verification/inputs",
    y_dir_path="data/auction+verification/targets",
    output_csv_sep=";",
):
    df = pd.read_csv(total_table_path)
    X_df = df[input_columns]
    y_df = df[output_columns]
    os.makedirs(X_dir_path, exist_ok=True)
    os.makedirs(y_dir_path, exist_ok=True)
    X_path = os.path.join(X_dir_path, X_filename)
    y_path = os.path.join(y_dir_path, y_filename)
    X_df.to_csv(X_path, sep=output_csv_sep)
    y_df.to_csv(y_path, sep=output_csv_sep)
    print("Inputs and targets exported into csv files")


def main():
    total_table_path = "data/auction+verification/data.csv"
    input_columns = [
        "process.b1.capacity",
        "process.b2.capacity",
        "process.b3.capacity",
        "process.b4.capacity",
        "property.price",
        "property.product",
        "property.winner",
    ]
    output_columns = ["verification.time"]
    X_filename = "inputs.csv"
    y_filename = "targets.csv"
    X_dir_path = "data/auction+verification/inputs"
    y_dir_path = "data/auction+verification/targets"
    output_csv_sep = ";"

    transform_total_table_to_input_and_target_tables(
        total_table_path=total_table_path,
        input_columns=input_columns,
        output_columns=output_columns,
        X_filename=X_filename,
        y_filename=y_filename,
        X_dir_path=X_dir_path,
        y_dir_path=y_dir_path,
        output_csv_sep=output_csv_sep,
    )


if __name__ == "__main__":
    main()
