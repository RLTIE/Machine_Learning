# Machine_Learning for Virtual Screening (VS)
# SVM Regression for Kinase Inhibition Prediction
This Python code performs Support Vector Machine (SVM) regression on kinase inhibition data obtained using a docking software. The code utilizes the scikit-learn library for SVM, feature selection, and model evaluation, as well as pandas, numpy, matplotlib, seaborn, and plotly for data manipulation, visualization, and analysis.
# Prerequisites
Ensure you have the following Python libraries installed:
numpy
pandas
matplotlib
plotly
scikit-learn
seaborn
# Data Loading
The kinase inhibition data is loaded from a CSV file named "full_data_kinases_labeled.csv" into a Pandas DataFrame.
# Data Preprocessing
The code includes normalization of the data using StandardScaler. Outliers are visualized and addressed through a function, although it is commented out as per the discussion about potential data manipulation.
# Feature Selection
Recursive Feature Elimination (RFE) is employed to select a subset of features. The code creates an SVM regression model with a linear kernel and applies RFE to select the most informative features.
# SVM Regression
The code builds an SVM regression model with a radial basis function (RBF) kernel using the selected features. It then evaluates the model on the test set, calculating Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Coefficient of Determination (R^2).
# Hyperparameter Tuning
GridSearchCV is employed to find the optimal hyperparameters for the SVM model. The best parameters and corresponding performance metrics are printed, and the best model is evaluated on the test set.
Principal Component Analysis (PCA)
Principal Component Analysis is performed on the features to visualize the data in a 2D scatter plot using Plotly Express.
# Usage
Ensure the required Python libraries are installed.
Provide the correct file path for your dataset in the line df = pd.DataFrame(pd.read_csv("full_data_kinases_labled.csv")).
Uncomment sections of code related to outlier removal, and different kernel function used in training the model if needed.
Run the code.
# Note
This README provides an overview of the code and its functionality. For detailed explanations, comments within the code should be referenced.
Author: Nour Abdollah
Date: 12/1/2024


