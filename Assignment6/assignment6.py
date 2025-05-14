import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load datasets
#https://www.kaggle.com/datasets/datamunge/sign-language-mnist
train_df = pd.read_csv("Assignment5/sign_mnist_train.csv")
test_df = pd.read_csv("Assignment5/sign_mnist_test.csv")

# Prepare features and labels
y_train = train_df.iloc[:, 0].to_numpy()
X_train = train_df.iloc[:, 1:].to_numpy()
y_test = test_df.iloc[:, 0].to_numpy()
X_test = test_df.iloc[:, 1:].to_numpy()

# Random Forest with Grid Search
rf_param_grid = {
    "n_estimators": [50],
    "max_depth": [10],
    "min_samples_split": [2, 5],
    "oob_score": [True],
    "n_jobs": [-1]
}

rf = RandomForestClassifier()
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, verbose=2, n_jobs=-1)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_

# Bagging with SVC and Grid Search
svc = SVC(kernel="linear")
bag_param_grid = {
    "n_estimators": [10],
    "max_features": [0.5],
    "bootstrap": [True]
}

bag_clf = BaggingClassifier(estimator=svc, oob_score=True, n_jobs=-1)
bag_grid = GridSearchCV(bag_clf, bag_param_grid, cv=5, verbose=2, n_jobs=-1)
bag_grid.fit(X_train, y_train)

best_bag = bag_grid.best_estimator_

# Model Evaluation
print(f"Random Forest Best Params: {rf_grid.best_params_}")
print(f"Bagging SVC Best Params: {bag_grid.best_params_}")

print(f"Random Forest - Train Score: {best_rf.score(X_train, y_train):.3f}")
print(f"Random Forest - Test Score: {best_rf.score(X_test, y_test):.3f}")
print(f"Random Forest - OOB Score: {best_rf.oob_score_:.3f}")

print(f"Bagging SVC - Train Score: {best_bag.score(X_train, y_train):.3f}")
print(f"Bagging SVC - Test Score: {best_bag.score(X_test, y_test):.3f}")
print(f"Bagging SVC - OOB Score: {best_bag.oob_score_:.3f}")

# Confusion Matrices
rf_cm = confusion_matrix(y_test, best_rf.predict(X_test), normalize="true")
bag_cm = confusion_matrix(y_test, best_bag.predict(X_test), normalize="true")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(rf_cm, display_labels=best_rf.classes_).plot(ax=ax[0])
ax[0].set_title("Random Forest")

ConfusionMatrixDisplay(bag_cm, display_labels=best_bag.classes_).plot(ax=ax[1])
ax[1].set_title("Bagging SVC")

plt.show()

# Feature Importances for Random Forest
importances = pd.DataFrame(
    best_rf.feature_importances_, index=train_df.columns[1:785], columns=["Importance"]
).sort_values(by="Importance", ascending=False)

importances.plot.bar(figsize=(10, 5), legend=False)
plt.title("Feature Importances in Random Forest")
plt.show()
