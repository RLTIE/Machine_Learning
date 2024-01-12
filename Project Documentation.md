Project Documentation
# Kinase Inhibition Prediction using SVM Regression
# Overview
This project utilizes Support Vector Machine (SVM) regression for predicting kinase inhibition based on data obtained from a docking software. The Python code employs various libraries, including NumPy, Pandas, Matplotlib, Plotly Express, Seaborn, and scikit-learn, to perform data preprocessing, feature selection, model building, hyperparameter tuning, and visualization.
# Project Structure
The code follows a structured approach with distinct sections for different tasks:
# Data Loading and Normalization:
The kinase inhibition data is loaded from the "full_data_kinases_labeled.csv" file into a Pandas DataFrame.
The data is normalized using StandardScaler to standardize feature values.
# Outlier Analysis and Handling:
A function is created to identify outliers based on the interquartile range (IQR).
A function to plot boxplots before and after outlier removal is included.
The option to remove outliers is discussed but commented out due to potential data manipulation concerns.
# Train-Test Split:
The dataset is split into training and testing sets using the train_test_split function in a ration 80:20.
# SVM Regression:
An SVM regression model with a linear kernel is created for the RFE.
Recursive Feature Elimination (RFE) is applied to select the most informative features.
Different kernel functions are used for training the model
# Model Evaluation:
The SVM model is trained on selected features and evaluated on the test set.
Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R^2) are calculated.
# Hyperparameter Tuning:
GridSearchCV is employed to find optimal hyperparameters for the SVM model.
The best parameters and corresponding performance metrics are printed.
# Principal Component Analysis (PCA):
PCA is performed to visualize the data in a 2D scatter plot using Plotly Express.
# Functions
Outlier Function:
outlier(df, feature): Identifies upper and lower limits for outlier detection based on the IQR.
Boxplot Function:
boxplot(df, feature): Generates a boxplot for a given feature to visualize data distribution.
Outlier Removal Function (Commented Out):
remove_outlier(df, feature): Removes outliers from the dataset, but it is commented out due to potential data manipulation concerns.
# Model Building and Evaluation
The code builds SVM regression models with different kernels (linear and radial basis function).
Feature selection is performed using RFE to enhance model interpretability.
Hyperparameter tuning is conducted through GridSearchCV to optimize model performance.
Model evaluation metrics, such as MSE, RMSE, and R^2, are calculated and reported.
# Visualization
Principal Component Analysis is applied to reduce dimensionality for data visualization.
A 2D scatter plot is created using Plotly Express to visualize the dataset after PCA.
# Dependencies
Ensure that the following Python libraries are installed:
NumPy
Pandas
Matplotlib
Plotly Express
scikit-learn
Seaborn
# Usage
Install the required libraries using pip install numpy pandas matplotlib plotly scikit-learn seaborn.
Provide the correct file path for your dataset in the line df = pd.DataFrame(pd.read_csv("full_data_kinases_labeled.csv")).
Uncomment sections of code related to outlier removal if needed.
Run the code.
# Note
This documentation provides an in-depth overview of the project's architecture, design decisions, algorithms used, and dependencies. For detailed explanations, comments within the code should be referenced.
Author: Nour
Date: 12/1/2024

