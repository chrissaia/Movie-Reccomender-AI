import pandas as pd
import os


def load_data(file_path: str) -> pd.DataFrame:     # assuming file is in csv format
    '''
    Loads data from csv file and returns it as a pandas dataframe
    :param file:
    :return: pd.DataFrame: Loaded data from csv file

    '''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File {file_path} not found.')

    return pd.read_csv(file_path)