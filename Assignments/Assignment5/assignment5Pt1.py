import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("fetal_health.csv")
columnname = 'baseline value'
#df = df[df[columnname] >= 5]
df = df.drop(columns= ['histogram_width','histogram_min','histogram_max','histogram_number_of_peaks',
                       'histogram_number_of_zeroes','histogram_mode','histogram_mean','histogram_median',
                       'histogram_variance','histogram_tendency','fetal_health'])

y = df[columnname].copy().to_numpy()
X = df.drop(columns=[columnname]).copy().to_numpy() 

# Train/test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
parameters = {"max_depth": range(2,16)}
grid_search = GridSearchCV(clf, param_grid= parameters, cv= 5)
grid_search.fit(X_train,y_train)
max_depth = grid_search.best_params_["max_depth"]
cld = DecisionTreeClassifier(max_depth= max_depth)


score_df = pd.DataFrame(grid_search.cv_results_)
#print(score_df.dtypes)
print(score_df[['param_max_depth', 'mean_test_score', 'rank_test_score']])

clf.fit(X_train, y_train)

print(f"Train Score: {clf.score(X_train, y_train):.3f}")
print(f"Test Score: {clf.score(X_test, y_test):.3f}")


cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

# Train a decision tree with max_depth=3
clf_depth_3 = DecisionTreeClassifier(max_depth=3)
clf_depth_3.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(12, 8))
plot_tree(clf_depth_3, feature_names=df.columns[1:], filled=True)
plt.title("Decision Tree (Max Depth = 3)")
plt.show()
