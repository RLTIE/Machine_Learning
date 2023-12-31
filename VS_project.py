# -*- coding: utf-8 -*-
"""VS_project.ipynb

Original file is located at
    https://colab.research.google.com/drive/17UCYNxwkGDU2NjKA84j9uvbhDT-M9REV

<img src="data:image/png;base64,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"
 style="float:right;width:50px;height:50px;">

<h1 align="center">Zewail University of Science and Technology
</h1>

<h1 align="center">Biomedical Sciences_ Computational Biology and Genomics_BMS 474</h1>
<h1 align="center">FALL 2023	</h1>
<h1 align="center">Project </h1>
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#the data was obtained using a docking software, both the label and the descriptors, but there is no python package for it
df = pd.DataFrame(pd.read_csv("full_data_kinases_labled.csv"))
df = df.rename(columns={'mean_dG': 'label'})
#print(df)
'''
normalizing the data
'''
Scaler = StandardScaler()
nor_df = pd.DataFrame(Scaler.fit_transform(df.iloc[:, :-1]))
nor_df.columns = df.columns[:-1]
#print(nor_df.head())

'''
building a function to remove outliers, the MSE if the model before removing the outliers is around 30
'''
def outlier(df, feature):
    Q1= df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    UpperLimit = Q3 + 1.5 * IQR
    LowerLimit = Q1 - 1.5* IQR
    return UpperLimit, LowerLimit

'''
building a fuction to plot the boxplot of the data to see the data  before and after removing the outliers
'''
def boxplot(df,feature):
    fig = plt.figure(figsize=(12,8))
    sb.boxplot(data = df,y = feature)
    plt.xlabel('label')
    plt.ylabel(f'{feature.name}')
    plt.title(f'Boxplot of {feature.name}')
    #plt.show()
    return
'''
this is to plot the boxplot of the data before removing outliers
'''
# for i in range(df.iloc[:,:-1].shape[1]):
#   boxplot(df,df.iloc[:,i])
'''
removing outliers, but as discussed with Dr. Eman during the presentation, this may manipulate the data so it will not be used, although the MSE was reduced to 21 
'''
def remove_outlier(df, feature):

  for i in range(df[feature].shape[0]):
    if df[feature].iloc[i] < outlier(df,feature)[1] or df[feature].iloc[i] > outlier(df,feature)[0]:
      df.loc[i,feature] = df[feature].median()
  return df

'''
#to remove outliers
'''
# for i in range(df.iloc[:,:-1].shape[1]):
#   remove_outlier(df, df.iloc[:,i].name)

'''
to plot the boxplot to see the effect of removing the outliers
'''
# for i in range(df.iloc[:,:-1].shape[1]):
#   boxplot(df,df.iloc[:,i])


'''
# Split the dataset into training and testing sets
'''
X_train, X_test, y_train, y_test = train_test_split(nor_df,df.iloc[:, -1], test_size=0.2, random_state=42)
#print(X_train, X_test, y_train, y_test)

'''
# Creating  an SVM classifier, an SVR specifically for regression 
'''
model_linear = SVR(kernel="linear")
'''
# Creating an RFE model and selecting 110, 50, and 25 of the most informative features. The best R2 score was obtained when using only 25 features
'''
model_linear.fit(X_train, y_train)
# rfe = RFE(estimator=model_linear, step = 1, n_features_to_select=110)
# rfe = RFE(estimator=model_linear, step = 1, n_features_to_select=50)
rfe = RFE(estimator=model_linear, step = 1, n_features_to_select=25)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
'''
# Training SVM model on selected features using different kernel functions
'''
# model_rbf = SVR(kernel="poly", degree=4)
# model_rbf = SVR(kernel="sigmoid")
# model_rbf = SVR(kernel="linear")
model_rbf = SVR(kernel="rbf")
model_rbf.fit(X_train_rfe, y_train)

'''
# Making predictions on the test set
'''
pred = model_rbf.predict(X_test_rfe)
#print(pred)

'''
# Calculate the mean squared error (MSE)
'''
mse = mean_squared_error(y_test, pred)
print(f"Mean Squared Error: {mse:.2f}")

'''
Calculate and print the root mean squared error (RMSE)
'''
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")
'''
# Printing the selected features
'''
selected_features = [i for i in range(len(rfe.support_)) if rfe.support_[i]]
print(f"Selected features: {selected_features}")
print (df.iloc[:, selected_features].columns)
'''
# Defining the parameter grid for multiple kernel functions 
'''
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
              'kernel': ['rbf']}
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
#               'kernel': ['sigmoid']}
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
#               'kernel': ['poly'], 'degree': [2,3,4,5]}
'''
# Creating the GridSearchCV object
'''
scorer = make_scorer(r2_score)
grid_search = GridSearchCV(model_rbf, param_grid, cv=5, scoring=scorer)

'''
# Fiting model on training data with selected features
'''
grid_search.fit(X_train_rfe, y_train)

'''
# Printing all results
'''
results_df = pd.DataFrame(grid_search.cv_results_)
print(results_df)
'''
# Printing the best parameter and its corresponding MSE, RMSE, R2
'''
bestC = grid_search.best_params_['C']
#bestdegree = grid_search.best_params_['degree']

best_r2 = grid_search.best_score_
print(f"Best C : {bestC}")
#print(f"Best degree : {bestdegree}")
print(f"Best Cross-Validation r2': {best_r2:.2f}")
'''
# Evaluate the model on the test set
'''

best_svm = grid_search.best_estimator_
y_pred_test = best_svm.predict(X_test_rfe)
mse = mean_squared_error(y_test, y_pred_test)
print(f"Mean Squared Error: {mse:.2f}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.2f}")
r2 = r2_score(y_test, y_pred_test)
print(f"Coefficient of Determination (R^2): {r2:.4f}")

'''
# Performing PCA on the features to visualize the data
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
pca_df = pd.DataFrame(pca.fit_transform(nor_df))
pca_df.index  = df.index
pca_df['label']  = df['label']
pca_df.head()
'''
'''

# Create a 2D scatter plot
fig = px.scatter(pca_df, x=0, y=1, color='label',
                 labels={0: 'PC 1', 1: 'PC 2', 'label': 'Label'})
'''
# Customize the plot
'''
fig.update_layout(title='2D Scatter Plot', xaxis_title='PC 1', yaxis_title='PC 2')

'''
# Show the plot
'''
fig.show()

