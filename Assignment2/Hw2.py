import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import numpy as np

# Load in cleaned and normalized March Madness data
df = pd.read_csv("cleaned_data.csv")

# Store all the features
X = df.iloc[:, 2:-1].to_numpy()
# Store the labels
y = df.iloc[:, -1].to_numpy()

print(f"Absolute largest feature value: {np.max(np.abs(X))}")
print(f"Absolute largest label value: {np.max(np.abs(y))}")

# Test size
test_size = 0.3

print("\n\n==== 20 random Train/Test splits fit with Linear Regression and Ridge ====")
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    reg = LinearRegression().fit(X_train, y_train)
    arg_max = np.argmax(np.abs(reg.coef_))
    rmse = np.average((reg.predict(X_test)-y_test)**2.0)**0.5
    print(f"Linear: R^2 = {reg.score(X_test, y_test):.3f}, RMSE = {rmse:.3f}, largest weight occurred at feature {arg_max} ({df.columns[arg_max+2]}) with value {reg.coef_[arg_max]}")
    reg = Ridge(alpha=1.0).fit(X_train, y_train)
    arg_max = np.argmax(np.abs(reg.coef_))
    rmse = np.average((reg.predict(X_test) - y_test) ** 2.0) ** 0.5
    print(f"Ride: R^2 = {reg.score(X_test, y_test):.3f}, RMSE = {rmse:.3f}, largest weight occurred at feature {arg_max} ({df.columns[arg_max+2]}) with value {reg.coef_[arg_max]}")
    print("-" * 120)

print("Finished")

