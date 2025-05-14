import numpy as np
import pandas as pd
from MyRegressionLibrary import *
from sklearn.linear_model import LinearRegression

from Testing import *

# Load and clean powerplant Dataset
df = pd.read_csv("Assignment1/powerplants.csv")
df_cleaned = df.dropna(subset=["capacity in MW", "generation_gwh_2021", "estimated_generation_gwh_2021", "latitude"])

X = df_cleaned[["capacity in MW", "generation_gwh_2021"]]
y = df_cleaned["estimated_generation_gwh_2021"]

reg = LinearRegression().fit(X, y)
print(f"Sklearn Linear Regression Coefficients: {reg.coef_}")
print(f"Sklearn Model R² Score: {reg.score(X, y)}")
print(f"Sklearn Model RMSE: {np.sqrt(np.average((y - reg.predict(X)) ** 2.0))}")

reg = MyRegressionLibrary().fit(X, y)
print(f"MyRegressionLibrary Coefficients: {reg.coef_}")
print(f"MyRegressionLibrary Model R² Score: {reg.score(X, y)}")
print(f"MyRegressionLibrary Model RMSE: {reg.RMSE(X, y)}")

plot_country_map("USA", "United States of America", df_cleaned)
#plot_country_map("CHN", "China", df_cleaned)