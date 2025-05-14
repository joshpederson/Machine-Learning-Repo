import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("fetal_health.csv")
df = df.drop(columns= ['histogram_width','histogram_min','histogram_max','histogram_number_of_peaks',
                       'histogram_number_of_zeroes','histogram_mode','histogram_mean','histogram_median',
                       'histogram_variance','histogram_tendency','fetal_health'])
columnname = 'baseline value'
y = df[columnname].copy().to_numpy()  # Select as the target variable
X = df.drop(columns=[columnname]).copy().to_numpy()  # Use all other columns as features

# Train/test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

unique, counts = np.unique(y_train, return_counts=True)
print(dict(zip(unique, counts)))