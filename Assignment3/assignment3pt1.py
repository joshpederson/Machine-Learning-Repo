import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datashader.datashape import to_numpy

import SVCPlot

# Load the dataset
df = pd.read_csv("cleaned_data.csv")

# # Update column names
# df.columns = [
#     "RA_ICRS", "DE_ICRS", "Source", "e_RA_ICRS", "e_DE_ICRS", "Plx", "e_Plx", "PM", "pmRA", "e_pmRA", "pmDE", "e_pmDE",
#     "RUWE", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "GRVSmag", "e_GRVSmag", "RV", "logg", "[Fe/H]",
#     "Dist", "PQSO", "PGal", "Pstar", "PWD", "Pbin", "Teff", "A0", "AG", "ABP", "ARP", "E(BP-RP)", "GMAG", "Rad",
#     "Rad-Flame", "Lum-Flame", "Mass-Flame", "Age-Flame", "z-Flame", "Evol", "EWHa", "e_EWHa", "f_EWHa"
# ]

# Convert int64 to float64
df[df.select_dtypes('int64').columns] = df.select_dtypes('int64').astype('float64')

# Selected features
X = df[[
    "RA_ICRS", "DE_ICRS", "e_RA_ICRS", "e_DE_ICRS", "Plx", "e_Plx", "PM", "pmRA", "e_pmRA", "pmDE", "e_pmDE",
    "RUWE", "Gmag", "e_Gmag", "BPmag", "e_BPmag", "RPmag", "e_RPmag", "GRVSmag", "e_GRVSmag", "RV", "logg", "[Fe/H]",
    "Dist", "PQSO", "PGal", "Pstar", "PWD", "Pbin", "Teff", "A0", "AG", "ABP", "ARP", "E(BP-RP)", "GMAG", "Rad",
    "Rad-Flame", "Lum-Flame", "Mass-Flame", "Age-Flame", "z-Flame", "Evol", "EWHa", "e_EWHa", "f_EWHa"
]].copy().to_numpy()

# This is the label
y = df["Source"].copy().to_numpy()

# Subtract avg and divide by std. deviation
X -= np.average(X, axis=0)
X /= np.std(X, axis=0)
y -= np.average(y)
y /= np.std(y)

# Train/Test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9)

# Set regression to lasso (or ridge)
reg = Lasso()
# Setting alpha to various numbers for training
# Returns "num" evenly spaced samples from start to stop
parameters = {"alpha": np.linspace(0.1, 0.1, num=10)}

# Grid search uses k-fold validation to find the optimal alpha
grid_search = GridSearchCV(reg, param_grid=parameters, cv=5, scoring="r2")  # Score with r2 like other regressions
grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)
print(score_df[['param_alpha', 'mean_test_score', 'rank_test_score']])

# nsmallest: bring best alpha to top row, iloc[0]: isolate top row, ['param_alpha']: get val from this col
optAlpha = round((score_df.nsmallest(1, 'rank_test_score')).iloc[0]['param_alpha'], 2)
reg.alpha = optAlpha

print()
print(f"----Final Testing W/ Optimal Alpha: {reg.alpha}----")
reg.fit(X_train, y_train)  # A final train on the whole training set all at once
print(f"R Squared Score: {reg.score(X_test, y_test)}")
print()