import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import MultiComparison

# NOTE please use the dataset provided with this submission.

DATA_FILEPATH = "Employee_Performance.csv"
COLOUR_SET = ['#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

def load_data(file_path):
    #seniority level was assigned using this excel formula:
    # =IF(D2>=8,"senior", IF(D2>=5, "mid-level", IF(D2>=2, "junior", "entry-level")))
    return pd.read_csv(file_path)


def print_summary(dataframe):
    print(dataframe.head())
    print(dataframe.describe())


## Check for outliers, missing values, and duplicates.
## Author - Skye
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


## Remove suspected outliers
## Author - Skye
def remove_outliers(dataframe, threshold = 1.5):
    q1 = dataframe['Salary'].quantile(0.25)
    q3 = dataframe['Salary'].quantile(0.75)
    iqr = q3 - q1

    # Identify rows outside threshold
    outlier_rows = dataframe[dataframe['Salary'] > q3 + (threshold * iqr)]
    return dataframe.drop(outlier_rows.index)


## Create plots for the categorical variables.
## Author - Skye
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
    plt.pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=90, colors=COLOUR_SET)
    plt.title('Proportion of Employee Count by Department', fontweight='bold')
    plt.show()

    # Graph proportion of departments
    seniority_counts = dataframe['Seniority'].value_counts()
    print(seniority_counts)
    plt.figure(figsize=(7, 7))
    plt.pie(seniority_counts, labels=seniority_counts.index, autopct='%1.1f%%', startangle=90, colors=COLOUR_SET)
    plt.title('Proportion of Employee Count by Seniority', fontweight='bold')
    plt.show()


## Create a histogram with a box plot stacked on top
## Author - Skye
def stacked_box_histplot(dataframe, column_name, x_label, y_label, title, bins=20, discrete=False):
    plt.figure(figsize=(8, 5))
    fig, (axis_box, axis_histogram) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
    sns.boxplot(dataframe[column_name], orient='h', ax=axis_box)
    sns.histplot(data=dataframe, x=column_name, bins=bins, ax=axis_histogram, discrete=discrete, kde=True)
    axis_box.set(xlabel='', title=title)
    axis_histogram.set(xlabel=x_label, ylabel=y_label)


## Create plots for the quantitative variables
## Author - Skye
def quantitative_plots(dataframe):
    # Years of experience
    plt.figure(figsize=(8,5))
    plt.title('Employees by Years of Experience', fontweight='bold')
    sns.set_style("whitegrid")
    sns.histplot(data=dataframe, x="Experience", bins=10, kde=False, discrete=True)
    plt.xticks([0,1,2,3,4,5,6,7,8,9])
    plt.xlabel('Years of Experience')
    plt.ylabel('Count of Employees')
    plt.show()

    # Training hours
    stacked_box_histplot(dataframe, 'TrainingHours',
                         'Hours of Training', 'Count of Employees',
                         'Employees by Training Hours', bins=10)
    plt.show()

    # Performance rating
    stacked_box_histplot(dataframe, 'PerformanceRating',
                         'Performance Rating', 'Count of Employees',
                         'Employees by Performance Rating', bins=10)
    plt.xticks([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5])
    plt.show()

    # Salary
    stacked_box_histplot(dataframe, 'Salary',
                         'Monthly Salary', 'Count of Employees', 'Employees by Salary Including Highly Paid Employees',
                         bins=10)
    plt.show()
    print(dataframe['Salary'].mode())


## Create a violin plot of performance rating grouped by seniority.
## Author - Skye
def plot_performance_by_experience(dataframe, department):
    violin = sns.boxplot(y=dataframe['PerformanceRating'], x=dataframe['Seniority'],
                         palette=COLOUR_SET, hue=dataframe['Seniority'])


    labels = ['Junior', 'Entry-Level', 'Mid-Level', 'Senior']
    violin.set_xticks(range(4))
    violin.set_xticklabels(labels)

    violin.set_xlabel('Seniority', fontweight='bold')
    violin.set_ylabel('Performance Rating')
    violin.set_title('Performance Rating by Experience in '+department+" Department", fontweight='bold')

    plt.show()


## Compare performance ratings between departments
## Author - Skye
def multivariate(dataframe):
    department_dataframes = {
        'IT': dataframe[dataframe['Department'] == 'IT'],
        'HR': dataframe[dataframe['Department'] == 'HR'],
        'Sales': dataframe[dataframe['Department'] == 'Sales'],
        'Marketing': dataframe[dataframe['Department'] == 'Marketing'],
    }
    for name, df in department_dataframes.items():
        plot_performance_by_experience(df, name)


## Create histograms with KDE to check for normality in performance ratings.
## Author - Skye
def graph_performance_normality(dataframe):
    plt.figure(figsize=(10, 6))
    departments = dataframe['Department'].unique()

    for i in range(len(departments)):
        plt.subplot(2,2, i+1)
        sns.histplot(data=dataframe[dataframe['Department'] == departments[i]], x='Experience', kde=True, bins=10)
        plt.title(f'Experience of {departments[i]} employees', fontweight='bold')
        plt.xlabel('Experience')
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


## Plot the correlation matrix of all quantitative variables.
## Author - Skye
def graph_correlation(dataframe):
    numerical_vars = ['Experience', 'TrainingHours', 'Salary', 'isMale', 'PerformanceRating']
    correlation_matrix = dataframe[numerical_vars].corr()

    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap for All Numerical Data', fontweight='bold')
    plt.show()


## Create scatter plots of the relationships between predictors and dependent
## Author - Skye
def graph_linearity(dataframe: pd.DataFrame, predictors: list[str], dependent: str):
    X = dataframe[predictors]
    Y = dataframe[dependent]

    fig, axes = plt.subplots(nrows=1, ncols=len(X.columns), figsize=(15, 5))

    for i, col in enumerate(X.columns):
        axes[i].scatter(X[col], Y, alpha=0.5)
        axes[i].set_title(f'{col} vs Performance Rating')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Performance Rating')

    plt.tight_layout()
    plt.show()


## Get an OLS linear regression model for the given dataset
## Author - Skye
def fit_linear_model(dataframe: pd.DataFrame, predictors: list[str], dependent: str) -> sm.regression.linear_model.RegressionResults:
    X = dataframe[predictors]
    Y = dataframe[dependent]

    X = sm.add_constant(X)
    return sm.OLS(Y, X).fit()


## Create qq plot of model residuals
def qq_plot(model):
    residuals = model.resid

    fig, ax = plt.subplots(figsize=(8, 4))
    sm.qqplot(residuals, line='s', ax=ax)
    plt.title("Q-Q Plot of Residuals")
    plt.show()


## Create homoscedasticity plot of residuals
## Author - Skye
def homoscedasticity_plot(model):
    residuals = model.resid
    predictions = model.fittedvalues

    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Predicted Values (Homoscedasticity Plot)")
    plt.show()


## Give the number of employees in each department
## Author - Gurleen
def count_departments(dataframe):
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


# Perform one-way ANOVA on performance ratings across different departments
# Author - Gurleen
def anova(dataframe):
    grouped_data = [dataframe[dataframe['Department'] == department]['PerformanceRating'] for department in
                    dataframe['Department'].unique()]

    # Perform ANOVA
    f_statistic, p_value = stats.f_oneway(*grouped_data)

    print(f'F-statistic: {f_statistic:.2f}')
    print(f'P-value: {p_value:.4f}')

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


## Run Tukey's post-hoc test to find which department is different
## Author - Gurleen
def tukeys_post_hoc(dataframe):
    # Perform Tukey's HSD post-hoc test
    multicomp = MultiComparison(dataframe['PerformanceRating'], dataframe['Department'])
    result = multicomp.tukeyhsd()

    print("\nTukey's HSD Post Hoc Test Results:")
    print(result.summary())


if __name__ == '__main__':
    # ~~~ Import Data ~~~ (Skye)
    dataframe = load_data(DATA_FILEPATH)
    print_summary(dataframe)
    check_data_validity(dataframe)
    # Not using remove_outliers
    # dataframe = remove_outliers(dataframe)

    # Create a boolean value from the gender field for use in multiple linear regression
    dataframe['isMale'] = [1 if gender == 'Male' else 0 for gender in dataframe['Gender']]

    # ~~~ Create Exploration Graphs ~~~ (Skye)
    categorical_plots(dataframe)
    quantitative_plots(dataframe)
    graph_performance_normality(dataframe)
    multivariate(dataframe)

    # ~~~ Hypothesis Testing ~~~ (Gurleen)
    count_departments(dataframe)
    anova(dataframe)
    tukeys_post_hoc(dataframe)

    # ~~~ Linear Regression ~~~ (Skye)
    # Transformation of experience doesn't work very well.
    dataframe['sqrtExperience'] = [math.log(10 * x) if x > 0 else 0 for x in dataframe['Experience']]
    graph_linearity(dataframe, ['Experience', 'sqrtExperience', 'TrainingHours', 'Salary'], 'PerformanceRating')
    graph_correlation(dataframe)

    model = fit_linear_model(dataframe, ['Experience', 'TrainingHours'], 'PerformanceRating')
    print(model.summary())
    qq_plot(model)
    homoscedasticity_plot(model)