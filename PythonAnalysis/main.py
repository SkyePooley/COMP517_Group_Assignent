import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import f
from statsmodels.stats.multicomp import MultiComparison

DATA_FILEPATH = "Employee_Performance.csv"
colour_set = ['#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

def load_data(file_path):
    #seniority level was assigned using this excel formula:
    # =IF(D2>=8,"senior", IF(D2>=5, "mid-level", IF(D2>=2, "junior", "entry-level")))
    return pd.read_csv(file_path)


def print_sumary(dataframe):
    print(dataframe.head())
    print(dataframe.describe())


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


def remove_outliers(dataframe, threshold = 1.5):
    q1 = dataframe['Salary'].quantile(0.25)
    q3 = dataframe['Salary'].quantile(0.75)
    iqr = q3 - q1

    # Identify rows outside threshold
    outlier_rows = dataframe[dataframe['Salary'] > q3 + (threshold * iqr)]
    return dataframe.drop(outlier_rows.index)


def categorical_plots(dataframe):
    # Graph proportion of gender
    gender_counts = dataframe['Gender'].value_counts()
    print(gender_counts)
    plt.figure(figsize=(5,5))
    plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Proportion of Male and Female Employees', fontweight='bold')
    plt.show()

    # Graph proportion of departments
    department_counts = dataframe['Department'].value_counts()
    print(department_counts)
    plt.figure(figsize=(7, 7))
    plt.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=90, colors=colour_set)
    plt.title('Proportion of Employee Count by Department', fontweight='bold')
    plt.show()

    # Graph proportion of departments
    seniority_counts = dataframe['Seniority'].value_counts()
    print(seniority_counts)
    plt.figure(figsize=(7, 7))
    plt.pie(seniority_counts, labels=seniority_counts.index, autopct='%1.1f%%', startangle=90, colors=colour_set)
    plt.title('Proportion of Employee Count by Seniority', fontweight='bold')
    plt.show()


def stacked_box_histplot(dataframe, column_name, x_label, y_label, title, bins=20, discrete=False):
    plt.figure(figsize=(8, 5))
    fig, (axis_box, axis_histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(dataframe[column_name], orient='h', ax=axis_box)
    sns.histplot(data=dataframe, x=column_name, bins=bins, ax=axis_histogram, discrete=discrete)
    axis_box.set(xlabel='', title=title)
    axis_histogram.set(xlabel=x_label, ylabel=y_label)


def quantitative_plots(dataframe):
    plt.figure(figsize=(8,5))
    plt.title('Employees by Years of Experience (excluding highly paid employees)', fontweight='bold')
    sns.set_style("whitegrid")
    sns.histplot(data=dataframe, x="Experience", bins=10, kde=False, discrete=True)
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.xlabel('Years of Experience')
    plt.ylabel('Count of Employees')
    plt.show()

    stacked_box_histplot(dataframe, 'TrainingHours',
                         'Hours of Training', 'Count of Employees',
                         'Employees by Training Hours')
    plt.show()

    stacked_box_histplot(dataframe, 'PerformanceRating',
                         'Performance Rating', 'Count of Employees',
                         'Employees by Performance Rating', bins=10)
    plt.xticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    plt.show()

    stacked_box_histplot(dataframe, 'Salary',
                         'Monthly Salary', 'Count of Employees', 'Employees by Salary Including Highly Paid Employees',
                         bins=30)
    plt.show()
    print(dataframe['Salary'].mode())


def plot_performance_by_experience(dataframe, department):
    violin = sns.violinplot(y=dataframe['PerformanceRating'], x=dataframe['Seniority'],
                            palette=colour_set, hue=dataframe['Seniority'])

    labels = ['Junior', 'Entry-Level', 'Mid-Level', 'Senior']
    violin.set_xticks(range(4))
    violin.set_xticklabels(labels)

    violin.set_xlabel('Seniority', fontweight='bold')
    violin.set_ylabel('Performance Rating')
    violin.set_title('Performance Rating by Experience in '+department+" Department", fontweight='bold')

    plt.show()


def multivariate(dataframe):
    department_dataframes = {
        'IT': dataframe[dataframe['Department'] == 'IT'],
        'HR': dataframe[dataframe['Department'] == 'HR'],
        'Sales': dataframe[dataframe['Department'] == 'Sales'],
        'Marketing': dataframe[dataframe['Department'] == 'Marketing'],
    }
    for name, df in department_dataframes.items():
        plot_performance_by_experience(df, name)



if __name__ == '__main__':
    dataframe = load_data(DATA_FILEPATH)
    print_sumary(dataframe)
    # check_data_validity(dataframe)
    # dataframe = remove_outliers(dataframe)
    # categorical_plots(dataframe)
    # quantitative_plots(dataframe)
    multivariate(dataframe)






def plot_kde_histograms(dataframe):
    # Set up the plot style
    sns.set(style="whitegrid")
    
    #unique departments
    departments = dataframe['Department'].unique()
    
    #figure for subplots
    plt.figure(figsize=(15, 10))
    
    # Loop through each department, KDE and histogram plot
    for i, department in enumerate(departments, 1):
        plt.subplot(2, 2, i)
        subset = dataframe[dataframe['Department'] == department]
        
        # KDE and Histogram plot for each department's performance rating
        sns.histplot(subset['PerformanceRating'], kde=True, bins=10, color=colour_set[i-1], edgecolor="black", alpha=0.7)
        plt.title(f'Performance Rating Distribution for {department} Department')
        plt.xlabel('Performance Rating')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

plot_kde_histograms(dataframe)

 

#Perform one-way ANOVA on performance ratings across different departments 
grouped_data = [dataframe[dataframe['Department'] == department]['PerformanceRating'] for department in dataframe['Department'].unique()]

# Perform ANOVA
f_statistic, p_value = stats.f_oneway(*grouped_data)

print(f'F-statistic: {f_statistic:.2f}')
print(f'P-value: {p_value:.4f}')


# Create an empty dictionary to store department counts
department_counts = {}

# Loop through unique department values
for department in dataframe['Department'].unique():
    # Count the occurrences of the current department and store it in the dictionary
    count = len(dataframe[dataframe['Department'] == department])
    department_counts[department] = count

# Print the department counts
for department, count in department_counts.items():
    print(f"Department {department}: {count} observations")


# Set significance level (alpha)
alpha = 0.05

# Perform interpretation based on p-value
if p_value < alpha:
    print("The performance ratings across different departments are significantly different.")
else:
    print("No significant difference in performance ratings among the departments.")

# Degrees of freedom
df_between = len(dataframe['Department'].unique()) - 1  # Number of groups - 1
df_within = len(dataframe) - len(dataframe['Department'].unique())  # Total samples - number of groups

# Calculate the critical F-value based on alpha and degrees of freedom
critical_f_value = stats.f.ppf(1 - alpha, df_between, df_within)

# Print results from ANOVA
print("One-way ANOVA Results:")
print(f"F-statistic: {f_statistic:.2f}")  # Use f_statistic from the ANOVA test
print(f"Critical F-value: {critical_f_value:.2f}")  # Critical F-value for comparison
print(f"P-value: {p_value:.4f}")  # P-value from ANOVA test

# Compare F-statistic to the critical F-value and make the decision
if f_statistic > critical_f_value:  # Use f_statistic instead of f_stat
    print("Reject the null hypothesis: there is a significant difference among departments.")
else:
    print("Fail to reject the null hypothesis: no significant difference among departments.")

print("---")


# Perform Tukey's HSD post-hoc test
multicomp = MultiComparison(dataframe['PerformanceRating'], dataframe['Department'])
result = multicomp.tukeyhsd()


print("\nTukey's HSD Post Hoc Test Results:")
print(result.summary())


if p_value < alpha:
    print("\nPost-hoc test confirms significant differences between department pairs.")
else:
    print("\nNo post-hoc test needed as ANOVA was not significant.")
