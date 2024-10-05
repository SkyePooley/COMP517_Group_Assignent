import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

DATA_FILEPATH = "Employee_Performance.csv"

def load_data(file_path):
    return pd.read_csv(file_path)

def check_data_validity(dataframe):
    print("~ Duplicate checks:")
    duplicates = dataframe[dataframe.duplicated(keep=False)]
    if duplicates.empty:
        print("There are no duplicates in the dataset.")
    else:
        print(duplicates)

    print("\n~ Missing value checks:")
    missing_count = dataframe.isnull().sum().sum()
    if missing_count == 0:
        print("There are no missing values in the dataset.")
    else:
        print("There are", missing_count, " missing values in the dataset.")

    print("\n~ Outlier Checks")
    numerical_columns = dataframe[['TrainingHours', 'Experience', 'PerformanceRating', 'Salary']]
    z_score_threshold = 3
    numerical_cols_z_scores = stats.zscore(numerical_columns)
    potential_outlier_mask = (numerical_cols_z_scores > z_score_threshold) | (numerical_cols_z_scores < -z_score_threshold)
    potential_outliers = numerical_columns[potential_outlier_mask]
    non_outliers = numerical_columns[~potential_outlier_mask]
    print("Potential outlier count: ", potential_outlier_mask.sum().sum())

    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
    fig.suptitle("Scatter plots with z scores over " + str(z_score_threshold) + " highlighted red X")
    for i, column in enumerate(numerical_columns.columns):
        axes[i].set_title(column)
        axes[i].set_ylabel(column)
        axes[i].set_xlabel('Index')
        axes[i].scatter(potential_outliers.index, potential_outliers[column], color='red', marker="x")
        axes[i].scatter(non_outliers.index, non_outliers[column], color='black')

    plt.show()

    sns.set_style("whitegrid")
    sns.histplot(data=dataframe, x="Salary", bins=20, kde=True)
    plt.show()



if __name__ == '__main__':
    rawDataframe = load_data(DATA_FILEPATH)
    check_data_validity(rawDataframe)
    print(rawDataframe.head())
    print(rawDataframe.describe())
    # TODO create and call methods to generate graphs and calculate statistics.
    # Avoid giving a method multiple jobs

