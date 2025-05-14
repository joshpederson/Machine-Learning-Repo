import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datashader.datashape import to_numpy
import SVCPlot

# https://www.kaggle.com/datasets/samanemami/credit-approval-loan

df = pd.read_csv("cleaned_file.csv")

df.dropna(inplace=True) # Drops all indices with a null


df = df.sample(1000) #Sample down dataset for run time

# Select variables
y = df["default payment next month"].copy().to_numpy()
X = df[["ID","LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]].copy().to_numpy()

#Normalize the data
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

#Train/Test split with 30% left as a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Set model to a Support Vector Classifier using linear kernel
clf = SVC(C = 1.0, kernel='linear')

#Setting C to various numbers for training
#Returns "num" evenly spaced sampled from start to stop
parameters = {'C': np.linspace(0.1, 10.0, num=10)}

#Grid search uses k-fold validation to find the optimal parameters
grid_search = GridSearchCV(clf, param_grid=parameters, cv=5) #cv = k = 5
grid_search.fit(X_train, y_train)
score_clf = pd.DataFrame(grid_search.cv_results_)
print(score_clf[['param_C', 'mean_test_score', 'rank_test_score']])

#Get the best performing C
#nsmallest: bring best alpha to top row, iloc[0]: isolate top row, ['param_alpha']: get val from this col
optC = round((score_clf.nsmallest(1, 'rank_test_score')).iloc[0]['param_C'], 2)
clf.C = optC

print()
print(f"----Final Testing W/ Optimal C: {clf.C}----")
clf.fit(X_train, y_train) #A final train on the whole training set all at once
print(f"Score: {clf.score(X_test, y_test):.6f}")

#Print confusion Matrix
cm = confusion_matrix(y_test, clf.predict(X_test), normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()

#print(y_test[:20])
#print(clf.predict(X_test))
