import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

# https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist
tmnist = pd.read_csv("TMNIST_Data.csv")

X = tmnist.loc[:, "1":"784"].copy().to_numpy()
y = tmnist["labels"].copy().to_numpy()

X.shape = (-1, 28, 28)

orientations = 8
ppc = 4

X_hog = np.zeros((X.shape[0], (28 // ppc) ** 2 * orientations))

canvas = np.zeros((280, 280))

for img in range(50):
    _, canvas[2 * (img // 10) * 28:2 * (img // 10) * 28 + 28, (img % 10) * 28: (img % 10) * 28 + 28] = hog(X[img],
                                                                                                           orientations=orientations,
                                                                                                           pixels_per_cell=(
                                                                                                           ppc, ppc),
                                                                                                           cells_per_block=(
                                                                                                           1, 1),
                                                                                                           feature_vector=True,
                                                                                                           visualize=True)
canvas /= np.max(canvas)

for img in range(50):
    canvas[2 * (img // 10) * 28 + 28:2 * (img // 10) * 28 + 28 + 28, (img % 10) * 28: (img % 10) * 28 + 28] = X[img] / 255.0
cv.imshow("HOGs", np.repeat(np.repeat(canvas, 3, axis=0), 3, axis=1))
cv.waitKey(0)
cv.destroyAllWindows()

for img in tqdm(range(X.shape[0]), desc="Computing HOGs"):
    X_hog[img] = hog(X[img], orientations=orientations, pixels_per_cell=(ppc, ppc), cells_per_block=(1, 1),
                     feature_vector=True)

X_hog = (X_hog - np.average(X_hog, axis=0)) / (np.std(X_hog, axis=0) + 1)

X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.3)

clf = SVC(C=50, kernel="rbf", decision_function_shape='ovo', gamma=0.125)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_train)

print(f"Score (Train): {accuracy_score(y_train, y_predict):.3f}")

y_predict = clf.predict(X_test)

print(f"Score (Test): {accuracy_score(y_test, y_predict):.3f}")
print(f"Number of support vectors: {clf.support_vectors_.shape[0]}")

cm = confusion_matrix(y_test, y_predict) #, normalize="true")
disp_cm = ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp_cm.plot()
plt.show()
