# Dependencies
import random
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Set the seed (reproducibility)
random.seed(0)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


# Data import and cleaning
df1 = pd.read_csv("./speedbumps_1.csv")  # read data from the .csv file
df2 = pd.read_csv("./speedbumps_2.csv")  # read data from the .csv file
df3 = pd.read_csv("./speedbumps_3.csv")  # read data from the .csv file
df4 = pd.read_csv("./speedbumps_4.csv")  # read data from the .csv file
df = pd.read_csv("./speedbumps_5.csv")  # read data from the .csv file
df1 = df1.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df2 = df2.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df3 = df3.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df4 = df4.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df = df.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
df = df.append(df1)
df = df.append(df2)
df = df.append(df3)
df = df.append(df4)
keywords = ['yes', 'no']
mapping = [1, 0]
df = df.replace(keywords, mapping)


# Random Forest Model 1
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('Speed', 'X', 'Y', 'Z', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = RandomForestClassifier(random_state=0)  # create a RandomForestClassifier
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

    # calculate precision
    precision = metrics.precision_score(test_Y, predicted_Y, average='binary', pos_label=1)
    precision = float(precision)

    # calculate recall
    recall = metrics.recall_score(test_Y, predicted_Y, average='binary', pos_label=1)
    recall = float(recall)

    # calculate F1 score
    if (precision + recall) != 0:
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)


# Calculate cross-validation average
print('\n-----------------------------------')
print('sklearn.ensemble.RandomForestClassifier Model 1')
print('\tFeatures: speed, X-accel, Y-accel, Z-accel, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))
print('\tZero F1 score:', f1_scores.count(0.00))


# # Random Forest Model 2
# # Separate Y and X variables
# df_label = df.loc[:, 'speedbump']
# df_feature = df.loc[:, ('Speed', 'X', 'Y', 'Z')]
# Y = df_label.as_matrix()
# X = df_feature.as_matrix()
#
#
# # Prepare for cross-validation
# clf = RandomForestClassifier(random_state=0)  # create a RandomForestClassifier
# f1_scores = []  # sum of F1 scores
# cv = 100  # number of cross-validations
#
#
# # Start cross-validation
# for i in range(0, cv, 1):
#
#     # split to train and test sets
#     train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)
#
#     # start training
#     clf = clf.fit(train_X, train_Y)  # fit the training data
#
#     # start testing
#     predicted_Y = clf.predict(test_X)  # predict on the testing data
#
#     # calculate precision
#     precision = metrics.precision_score(test_Y, predicted_Y, average='binary', pos_label=1)
#     precision = float(precision)
#
#     # calculate recall
#     recall = metrics.recall_score(test_Y, predicted_Y, average='binary', pos_label=1)
#     recall = float(recall)
#
#     # calculate F1 score
#     if (precision + recall) != 0:
#         f1 = (2 * precision * recall) / (precision + recall)
#         f1_scores.append(f1)
#
#
# # Calculate cross-validation average
# print('\n-----------------------------------')
# print('sklearn.ensemble.RandomForestClassifier Model 2')
# print('\tFeatures: speed, X-accel, Y-accel, Z-accel')
# print('\tLabels: speedbump (1 = yes, 0 = no)')
# print('\tAverage F1 score:', np.mean(f1_scores))
# print('\tStdDev F1 score:', np.std(f1_scores))
# print('\tMedian F1 score:', np.median(f1_scores))
# print('\tIQR F1 score:', stats.iqr(f1_scores))
# print('\tSkewness F1 score:', stats.skew(f1_scores))
# print('\tZero F1 score:', f1_scores.count(0.00))
#
#
# # Random Forest Model 3
# # Separate Y and X variables
# df_label = df.loc[:, 'speedbump']
# df_feature = df.loc[:, ('Speed', 'X', 'Y', 'z_jolt')]
# Y = df_label.as_matrix()
# X = df_feature.as_matrix()
#
#
# # Prepare for cross-validation
# clf = RandomForestClassifier(random_state=0)  # create a RandomForestClassifier
# f1_scores = []  # sum of F1 scores
# cv = 100  # number of cross-validations
#
#
# # Start cross-validation
# for i in range(0, cv, 1):
#
#     # split to train and test sets
#     train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)
#
#     # start training
#     clf = clf.fit(train_X, train_Y)  # fit the training data
#
#     # start testing
#     predicted_Y = clf.predict(test_X)  # predict on the testing data
#
#     # calculate precision
#     precision = metrics.precision_score(test_Y, predicted_Y, average='binary', pos_label=1)
#     precision = float(precision)
#
#     # calculate recall
#     recall = metrics.recall_score(test_Y, predicted_Y, average='binary', pos_label=1)
#     recall = float(recall)
#
#     # calculate F1 score
#     if (precision + recall) != 0:
#         f1 = (2 * precision * recall) / (precision + recall)
#         f1_scores.append(f1)
#
#
# # Calculate cross-validation average
# print('\n-----------------------------------')
# print('sklearn.ensemble.RandomForestClassifier Model 3')
# print('\tFeatures: speed, X-accel, Y-accel, Z-jolt')
# print('\tLabels: speedbump (1 = yes, 0 = no)')
# print('\tAverage F1 score:', np.mean(f1_scores))
# print('\tStdDev F1 score:', np.std(f1_scores))
# print('\tMedian F1 score:', np.median(f1_scores))
# print('\tIQR F1 score:', stats.iqr(f1_scores))
# print('\tSkewness F1 score:', stats.skew(f1_scores))
# print('\tZero F1 score:', f1_scores.count(0.00))
#
#
# # Random Forest Model 4
# # Separate Y and X variables
# df_label = df.loc[:, 'speedbump']
# df_feature = df.loc[:, ('X', 'Y', 'Z', 'z_jolt')]
# Y = df_label.as_matrix()
# X = df_feature.as_matrix()
#
#
# # Prepare for cross-validation
# clf = RandomForestClassifier(random_state=0)  # create a RandomForestClassifier
# f1_scores = []  # sum of F1 scores
# cv = 100  # number of cross-validations
#
#
# # Start cross-validation
# for i in range(0, cv, 1):
#
#     # split to train and test sets
#     train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, shuffle=True)
#
#     # start training
#     clf = clf.fit(train_X, train_Y)  # fit the training data
#
#     # start testing
#     predicted_Y = clf.predict(test_X)  # predict on the testing data
#
#     # calculate precision
#     precision = metrics.precision_score(test_Y, predicted_Y, average='binary', pos_label=1)
#     precision = float(precision)
#
#     # calculate recall
#     recall = metrics.recall_score(test_Y, predicted_Y, average='binary', pos_label=1)
#     recall = float(recall)
#
#     # calculate F1 score
#     if (precision + recall) != 0:
#         f1 = (2 * precision * recall) / (precision + recall)
#         f1_scores.append(f1)
#
#
# # Calculate cross-validation average
# print('\n-----------------------------------')
# print('sklearn.ensemble.RandomForestClassifier Model 4')
# print('\tFeatures: X-accel, Y-accel, Z-accel, Z-jolt')
# print('\tLabels: speedbump (1 = yes, 0 = no)')
# print('\tAverage F1 score:', np.mean(f1_scores))
# print('\tStdDev F1 score:', np.std(f1_scores))
# print('\tMedian F1 score:', np.median(f1_scores))
# print('\tIQR F1 score:', stats.iqr(f1_scores))
# print('\tSkewness F1 score:', stats.skew(f1_scores))
# print('\tZero F1 score:', f1_scores.count(0.00))


# Random Forest Model 5
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('Speed', 'Z', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = RandomForestClassifier(random_state=0)  # create a RandomForestClassifier
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

    # calculate precision
    precision = metrics.precision_score(test_Y, predicted_Y, average='binary', pos_label=1)
    precision = float(precision)

    # calculate recall
    recall = metrics.recall_score(test_Y, predicted_Y, average='binary', pos_label=1)
    recall = float(recall)

    # calculate F1 score
    if (precision + recall) != 0:
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)


# Calculate cross-validation average
print('\n-----------------------------------')
print('sklearn.ensemble.RandomForestClassifier Model 5')
print('\tFeatures: speed, Z-accel, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))
print('\tZero F1 score:', f1_scores.count(0.00))
