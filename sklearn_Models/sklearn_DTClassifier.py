# Dependencies
import random
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# Set the seed (reproducibility)
random.seed(0)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


# Data import and cleaning
df = pd.read_csv("./kenya_jolt_2.csv")  # read data from the .csv file
df = df.loc[:, ('speedbump', 'x', 'y', 'z', 'x_jolt', 'y_jolt', 'z_jolt')]  # only select relevant columns


# Decision Tree Model 1
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('x', 'y', 'z', 'x_jolt', 'y_jolt', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = DecisionTreeClassifier()  # create a DecisionTreeClassifier
f1_scores = []  # sum of F1 scores
cv = 100  # number of cross-validations


# Start cross-validation
for i in range(0, cv, 1):

    # split to train and test sets
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # start training
    clf = clf.fit(train_X, train_Y)  # fit the training data

    # start testing
    predicted_Y = clf.predict(test_X)  # predict on the testing data

    # calculate the F1 score
    f1 = metrics.f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
    f1_scores.append(f1)

    # calculate the confusion matrix
    matrix = metrics.confusion_matrix(test_Y, predicted_Y)


# Calculate cross-validation average
print('\n-----------------------------------')
print('sklearn.tree.DecisionTreeClassifier Model 1')
print('\tFeatures: X-accel, Y-accel, Z-accel, X-jolt, Y-jolt, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))


# Decision Tree Model 2
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('x', 'y', 'z', 'y_jolt', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = DecisionTreeClassifier()  # create a DecisionTreeClassifier
f1_scores = []  # sum of F1 scores
cv = 100  # number of cross-validations


# Start cross-validation
for i in range(0, cv, 1):

    # split to train and test sets
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # start training
    clf = clf.fit(train_X, train_Y)  # fit the training data

    # start testing
    predicted_Y = clf.predict(test_X)  # predict on the testing data

    # calculate the F1 score
    f1 = metrics.f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
    f1_scores.append(f1)

    # calculate the confusion matrix
    matrix = metrics.confusion_matrix(test_Y, predicted_Y)


# Calculate cross-validation average
print('\n-----------------------------------')
print('sklearn.tree.DecisionTreeClassifier Model 2')
print('\tFeatures: X-accel, Y-accel, Z-accel, Y-jolt, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))


# Decision Tree Model 3
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('y', 'z', 'x_jolt', 'y_jolt', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = DecisionTreeClassifier()  # create a DecisionTreeClassifier
f1_scores = []  # sum of F1 scores
cv = 100;  # number of cross-validations


# Start cross-validation
for i in range(0, cv, 1):

    # split to train and test sets
    train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)

    # start training
    clf = clf.fit(train_X, train_Y)  # fit the training data

    # start testing
    predicted_Y = clf.predict(test_X)  # predict on the testing data

    # calculate the F1 score
    f1 = metrics.f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
    f1_scores.append(f1)

    # calculate the confusion matrix
    matrix = metrics.confusion_matrix(test_Y, predicted_Y)


# Calculate cross-validation average
print('\n-----------------------------------')
print('sklearn.tree.DecisionTreeClassifier Model 3')
print('\tFeatures: Y-accel, Z-accel, X-jolt, Y-jolt, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))
