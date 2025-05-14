import numpy as np
import pandas as pd
from MyRegressionLibrary import *
from sklearn.linear_model import LinearRegression

from Testing import *

#load powerplant Dataset
df = pd.read_csv("powerplants.csv")
cdf = pd.read_csv("cleaned_powerplant_data.csv")
cdf1 = pd.read_csv("cleaned_powerplant_data1.csv")
#print(df.columns)

X = cdf1[["capacity in MW", "generation_gwh_2021"]]
y = cdf1["estimated_generation_gwh_2021"]


reg = LinearRegression().fit(X, y)
print(f"Sklearn Linear Regression Coefficients: {reg.coef_}")
print(f"Sklearn Model R² Score: {reg.score(X, y)}")
print(f"Sklearn Model RMSE: {np.sqrt(np.average((y - reg.predict(X)) ** 2.0))}")

reg = MyRegressionLibrary().fit(X, y)
print(f"MyRegressionLibrary Coefficients: {reg.coef_}")
print(f"MyRegressionLibrary Model R² Score: {reg.score(X, y)}")
print(f"MyRegressionLibrary Model RMSE: {reg.RMSE(X, y)}")



#print(cdf1[["capacity in MW", "generation_gwh_2021", "estimated_generation_gwh_2021"]].corr())

plot_country_map("USA", "United States of America", df)
#plot_country_map("CHN", "China", df)