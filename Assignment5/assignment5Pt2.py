import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv("milknew.csv")
df = df.drop(columns="Grade")

# Select variables
# Names pH,Temprature,Taste,Odor,Fat ,Turbidity,Colour,Grade
columnname = 'Taste'
y = df[columnname].copy().to_numpy()  # Select 'Taste' as the target variable
X = df.drop(columns=[columnname]).copy().to_numpy()  # Use all other columns as features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

parameters = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid_search_svc = GridSearchCV(SVC(), param_grid=parameters, cv=5, scoring='accuracy')
grid_search_svc.fit(X_train, y_train)
print(f"Best Parameters for SVC: {grid_search_svc.best_params_}")
print(f"Train Accuracy: {grid_search_svc.best_score_:.3f}")
svc_best = grid_search_svc.best_estimator_
print(f"Test Accuracy: {svc_best.score(X_test, y_test):.3f}")

parameters_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid=parameters_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

print(f"Best Parameters for Random Forest: {grid_search_rf.best_params_}")
print(f"Train Accuracy: {grid_search_rf.best_score_:.3f}")
rf_best = grid_search_rf.best_estimator_
print(f"Test Accuracy: {rf_best.score(X_test, y_test):.3f}")

