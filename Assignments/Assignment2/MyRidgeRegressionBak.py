#Create your own class for ridge regression using numpy and linear algebra.
#The class should include a fit, predict, score, and RMSE method.
#Repeat problem 2 with your own regression library and confirm that you get the same result. (2 points)

# Import the numpy library for math operations
import numpy as np

# Define the MultipleLinearRegression class
class MyRidgeRegressionBak:
    # Default constructor sets coefficients and intercept to none
    def __init__(self, lambda_ = 1.0):
        self.coef_ = None  # This will hold the coefficients (weights) for the features
        self.intercept_ = None  # This will hold the intercept (bias term)
        self.lambda_ = lambda_ # This will hold the regularization strength. Goes from 0->infinity.

    # The fit function calculates the coefficients and intercepts by training it on the entered data with
    # the features and labels
    def fit(self, X, y):
        #Making the mean of features and labels = 0 helps the regression model and eliminate col of 0s
        XMean = np.average(X, axis=0)
        yMean = np.average(y)
        X = (X - XMean) #TODO dividing by std. deviation ruins my R squared...
        y = (y - yMean)

        # We do not need to make a feature matrix M since we are centering the labels and
        # features. This eliminates the column of ones since B0 should be 0. However,
        # I keep in the feature matrix so that it multiplies with the y vector
        M = np.column_stack((np.ones(X.shape[0]), X))

        # beta is the vector of coefficients
        # For ridge regression B = ((XT*X + lambdaI)**-1)XT*y
        # np.linalg.inv() = inverse of a matrix, @ = matrix multiplication operator, .T = transpose a vector/matrix
        beta = np.linalg.inv(M.T @ M + self.lambda_ * np.identity(M.T.shape[0])) @ M.T @ y

        self.coef_ = beta[1:]

        # The intercept attempts to add back y_mean and subtracts coef*X_mean to account for the centered feature values
        self.intercept_ = yMean - self.coef_ @ XMean #readjusts bias to original scale

        return self

    # Using the model trained from the "fit" function, you can enter features to see what the predicted label values
    # are. Multiplies X matrix of features with the coefficient (beta vector) + intercept
    def predict(self, X):
        return X @ self.coef_ + self.intercept_  # Return the predicted values

    # The score function calculates how well the model performs by returning the R-squared score, which is one of
    # the ways to score the accuracy of a model. This shows how much of the variance is explained by the model.
    def score(self, X, y):
        # np.sum() adds all elements in the array, and np.average() calculates the average of the elements.
        # This returns R squared score = 1 - ((summation(yi - yhati)**2)/(summation(yi-ymean)**2))
        return 1.0 - np.sum((y - self.predict(X)) ** 2.0) / np.sum((y - np.average(y)) ** 2.0)

    # The RMSE function calculates the Root Mean Squared Error, which is another method of scoring the accuracy
    # of a model. It is the average error between predicted and actual values.
    def RMSE(self, X, y):
        # np.sqrt() calculates the square root, and np.average() computes the mean of the squared errors.
        #This returns RMSE = sqrt((1/N)summation(yhati-yi)**2)
        return np.sqrt(np.average((self.predict(X) - y) ** 2.0))
