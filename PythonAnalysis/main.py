import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILEPATH = "Employee_Performance.csv"

def load_data(file_path):
    return pd.read_csv(file_path)

def clean_data(dataframe):
    # TODO check dataset for missing values, outliers, and duplicates. Return clean copy.
    return dataframe


if __name__ == '__main__':
    rawDataframe = load_data(DATA_FILEPATH)
    cleanDataframe = clean_data(rawDataframe)
    print(cleanDataframe.head())
    # TODO create and call methods to generate graphs and calculate statistics.
    # Avoid giving a method multiple jobs

