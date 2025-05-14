Assignments from Machine Learning

Assignment 1:

Assignment Details:



Sklearn Linear Regression Coefficients: [3.05932727 0.29200699]

Sklearn Model R² Score: 0.9782106022762582

Sklearn Model RMSE: 122.76096517021871

MyRegressionLibrary Coefficients: [3.05932727 0.29200699]

MyRegressionLibrary Model R² Score: 0.9782106022762582

MyRegressionLibrary Model RMSE: 122.76096517021875

These are the weights learned by my custom linear regression model. The first value (≈3.06) is the coefficient for "capacity in MW", and the second value (≈0.29) is for "generation_gwh_2021". This means the model predicts "estimated_generation_gwh_2021" as a linear combination of these two features.

The R² score (coefficient of determination) indicates how well the model explains the variance in the target variable. A score of 0.978 means my model explains about 97.8% of the variance, which is very high and suggests a good fit.

The Root Mean Squared Error (RMSE) measures the average prediction error in the same units as the target variable. Here, the average error is about 123 GWh.

The code selects "capacity in MW" and "generation_gwh_2021" as the input features and "estimated_generation_gwh_2021" as the target variable for regression analysis.
The script fits a linear regression model using scikit-learn’s LinearRegression class and prints the model’s coefficients, R² score, and root mean squared error (RMSE).
It then fits another linear regression model using a custom regression library called MyRegressionLibrary and prints the corresponding coefficients, R² score, and RMSE.
Finally, the script visualizes the power plants in the United States by calling the plot_country_map function, which displays the locations and primary energy resources of the power plants on an interactive map.

This code cleans and models power plant data, compares two regression implementations, and visualizes power plants and their fuel mix on an interactive country map. The nearly identical results between scikit-learn and my custom library. This comfirms that my custom implementation is correct.


~
Assignment 6: 

Assignment Details:
Find a classification dataset that you have not used before (this means you cannot not use the standard MNIST or the TMNIST datasets). There should be at least three distinct categorical labels, but only quantitative features, or features which can have an indicator variable, i.e., coded as either 1 or 0. Use a grid search to find the optimal parameters for a Random Forest. Repeat the process for a Bagging Classifier with a Linear Support Vector Classifier. Compare the test score with the OOB Score for each. Are they similar? Train on the entire dataset and compare the results of the Bagging SVC and the Random Forest. Be ready to explain the strengths and weaknesses of each as they relate to your chosen dataset.

The script loads the Sign Language MNIST training and test datasets from CSV files. The first column is used as the label, and the remaining columns are used as pixel features.
It extracts the labels and features from both the training and test datasets, converting them into NumPy arrays for use with scikit-learn models.
The script defines a grid of hyperparameters for the Random Forest classifier, including the number of estimators, maximum tree depth, and minimum samples required to split a node. It uses GridSearchCV to perform cross-validated grid search and fits the best model to the training data.
A Bagging classifier is created using a linear SVC as the base estimator. The script defines a grid of hyperparameters for the Bagging classifier and uses GridSearchCV to find the best combination. The best Bagging model is then fit to the training data.
The script prints the best hyperparameters found for both the Random Forest and Bagging SVC models. It evaluates both models on the training and test sets, printing their accuracy scores and out-of-bag (OOB) scores.
Confusion matrices are generated for both models using the test set predictions. The script displays these matrices side by side using matplotlib, allowing for a visual comparison of classification performance.
The script calculates feature importances from the best Random Forest model. It creates a bar plot showing the importance of each pixel feature, helping to identify which pixels are most influential in the classification task.


Random Forest - Train Score: 0.988

The Random Forest model correctly classifies 98.8% of the training data, indicating it fits the training set very well.

Random Forest - Test Score: 0.703

The model only achieves 70.3% accuracy on the test data, which is much lower than on the training set. This suggests the model does not generalize well to new, unseen data.

Random Forest - OOB Score: 0.944

The out-of-bag (OOB) score is 94.4%, which is an internal cross-validation score for Random Forests. It is much higher than the test score, indicating possible overfitting or that the OOB samples are not representative of the test set.

Bagging SVC - Train Score: 1.000

The Bagging SVC model achieves perfect accuracy on the training data, which is a strong sign of overfitting.

Bagging SVC - Test Score: 0.790

The test accuracy is 79.0%, which is significantly lower than the training accuracy, again indicating overfitting.

Bagging SVC - OOB Score: 0.991

The OOB score is 99.1%, which is much higher than the test score and suggests the model is not generalizing well.

To fix these issues I could try:

For Random Forest, try lowering max_depth, increasing min_samples_split, or reducing n_estimators to make the model less complex.

For Bagging, Reduce the number of estimators

Expand the grid search to include a wider range of values for parameters like max_depth, min_samples_split, and n_estimators.

For SVC, try different kernels and regularization parameters (C).

![ConfusionMaxtrixAssignment6](https://github.com/user-attachments/assets/f0485068-3fb4-471e-a94e-f887b796c89f)

![FeatureImportanceAssignment6](https://github.com/user-attachments/assets/1a04173e-ddda-4de5-9cf6-a65483720890)


~

Assignment 7: Clustering

Assignment Details: Do something creative with Clustering

This script analyzes and visualizes U.S. power plant data by clustering plants geographically and summarizing each cluster. It also generates an interactive map for exploration.

The script reads a CSV file of U.S. power plants and drops any rows missing the primary source, coordinates, or capacity information. It computes the great-circle distance from each plant to Menomonie, WI. Power plants are clustered geographically using DBSCAN with haversine distance. For each cluster, the script calculates the total capacity, number of plants, average distance to the cluster center, and counts of each primary source. It also finds the plant closest to Menomonie, exports a CSV summary and plant-level data for each cluster in a clusters_output folder, and prints a summary to the console. Finally, the script creates a Folium map showing all plants colored by primary source, adds a marker for Menomonie, WI, includes a legend for energy sources, and saves the map as powerplant_clusters.html if it does not already exist.

Dataset:

https://atlas.eia.gov/datasets/eia::power-plants/

See powerplant_clusters.html for intereractive map of powerplants in the US

~

Assignment 8: MNIST Digit Classifier

Assignment Details: Do something creative with NN

This code loads the MNIST data set and Runs a CNN that has 3 convolutional layers and 2 max pool layers. It trains the the CNN with an output of ~97%. Then it visualized what happends at each step with a images chosen in the code. For this example I chosed the very first image in the dataset.

Convolutional Layers (Conv): Extract and learn hierarchical features from the input images.

Dropout (Drop): Regularizes the network to prevent overfitting.

Max Pooling (Max pool): Downsamples feature maps, making the model more efficient and invariant to small shifts in the input.
		
The dataset is too large for github so please download the dataset from the link below

Data set: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

Step 1: Convolution1	
![Conv1](https://github.com/user-attachments/assets/c2ce5bbf-eeb5-4420-b142-a14ec5c6b0bc)

Step 2: Convolution2 + Dropout
![Conv2+Dropout](https://github.com/user-attachments/assets/90d604d0-00a6-41e6-8434-3a2c3f10707c)

Step 3: Max Pool 1
![Maxpool1](https://github.com/user-attachments/assets/2ab65176-13c9-4ca7-b8ab-80f0e7cd9e93)

Step 4: Convolution3
![Conv3](https://github.com/user-attachments/assets/ab9aad47-583a-4113-86b8-9e4ff480b2c6)

Step 5: Max Pool 3
![MaxPool2](https://github.com/user-attachments/assets/f3ec9def-ce41-4d12-a463-ddf3b070d14d)
