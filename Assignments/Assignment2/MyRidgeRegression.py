import numpy as np

class MyRidgeRegression:
    def __init__(self, alpha=1.0, fit_intercept=True):
        """
        Initialize the Ridge Regression model with regularization parameter alpha
        and an option to fit an intercept.
        """
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """
        Fit the Ridge Regression model to the training data.
        
        If fit_intercept is True, add a column of ones to X to account for the intercept.
        Create an identity matrix I for regularization, but do not regularize the intercept term.
        Compute the Ridge Regression coefficients using the normal equation.
        Separate the intercept from the coefficients if fit_intercept is True.
        """
        if self.fit_intercept:
            # Add a column of ones to X for the intercept term
            X = np.column_stack((np.ones(X.shape[0]), X))

        # Create an identity matrix with the same number of columns as X
        I = np.eye(X.shape[1])
        if self.fit_intercept:
            # Do not regularize the intercept term
            I[0, 0] = 0

        # Compute the Ridge regression coefficients
        self.coef_ = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

        if self.fit_intercept:
            # Separate the intercept from the coefficients
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0

    def predict(self, X):
        """
        Predict the target values using the Ridge Regression model.
        
        If fit_intercept is True, add the intercept to the predictions.
        """
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        return X @ self.coef_

    def score(self, X, y):
        """
        Calculate the R^2 score of the model, which is a measure of how well
        the model's predictions match the true values.
        
        Compute the total sum of squares (ss_total) and the residual sum of squares (ss_residual).
        The R^2 score is 1 minus the ratio of ss_residual to ss_total.
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)

    def rmse(self, X, y):
        """
        Calculate the Root Mean Squared Error (RMSE) of the model's predictions,
        which is a measure of the average difference between the predicted and true values.
        """
        y_pred = self.predict(X)
        return np.sqrt(np.mean((y - y_pred) ** 2))
