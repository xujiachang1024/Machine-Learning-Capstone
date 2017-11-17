# Dependencies
import random
import pandas as pd
import numpy as np
import warnings
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics


# Set the seed (reproducibility)
random.seed(0)
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


# Data import and cleaning
df = pd.read_csv("./kenya_oct_15_data_labeled.csv")  # read data from the .csv file


# Decision Tree Model 1
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('x', 'z')]

numPoints = 4;
feature1 = "x"
feature2 = "z"
feature3 = ""
feature4 = ""
feature5 = ""
feature6 = ""

for i in range(1,numPoints+1):
    if feature1 != "":
        new_feature = (feature1, i)
        df_feature[new_feature] = df_feature[feature1].shift(i)
    if feature2 != "":
        new_feature = (feature2, i)
        df_feature[new_feature] = df_feature[feature2].shift(i)
    if feature3 != "":
        new_feature = (feature3, i)
        df_feature[new_feature] = df_feature[feature3].shift(i)
    if feature4 != "":
        new_feature = (feature4, i)
        df_feature[new_feature] = df_feature[feature4].shift(i)
    if feature5 != "":
        new_feature = (feature5, i)
        df_feature[new_feature] = df_feature[feature5].shift(i)
    if feature6 != "":
        new_feature = (feature6, i)
        df_feature[new_feature] = df_feature[feature6].shift(i)

df_feature = df_feature[numPoints:]
df_label = df_label[numPoints:]
df_feature.index = range(len(df_feature))
df_label.index = range(len(df_label))

Y = df_label[:13000].as_matrix()
X = df_feature[:13000].as_matrix()

# Prepare for cross-validation
clf = ExtraTreesClassifier(random_state=0)  # create a MLPClassifier
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
print('sklearn_Models.ensemble.ExtraTreeClassifier Model 1')
print('\tFeatures: speed, X-accel, Y-accel, Z-accel, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))

x_test = df_feature[13000:]
y_test = df_label[13000:]
x_test.index = range(len(x_test))
y_test.index = range(len(y_test))

predictions = clf.predict(x_test)
ser = pd.Series(predictions)
ser.name = "predictions"
print(ser.value_counts())
print('\n-----------------------------------')
print(y_test.value_counts())
output = pd.concat([y_test, ser], axis=1)
output.to_csv(path_or_buf='output.csv')



# predictions = clf.predict(df_feature)
# ser = pd.Series(predictions)
# ser.name = "predictions"
# print(ser.value_counts())
# print('\n-----------------------------------')
# print(df_label.value_counts())
# output = pd.concat([df_label, ser], axis=1)
# output.to_csv(path_or_buf='output.csv')


# # Extra Tree Model 2
# # Separate Y and X variables
# df_label = df.loc[:, 'speedbump']
# df_feature = df.loc[:, ('Speed', 'X', 'Y', 'Z')]
# Y = df_label.as_matrix()
# X = df_feature.as_matrix()
#
#
# # Prepare for cross-validation
# clf = ExtraTreesClassifier(random_state=0)  # create a MLPClassifier
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
#     # calculate the F1 score
#     f1 = metrics.f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
#     f1_scores.append(f1)
#
#     # calculate the confusion matrix
#     matrix = metrics.confusion_matrix(test_Y, predicted_Y)
#
#
# # Calculate cross-validation average
# print('\n-----------------------------------')
# print('sklearn_Models.ensemble.ExtraTreeClassifier Model 2')
# print('\tFeatures: speed, X-accel, Y-accel, Z-accel')
# print('\tLabels: speedbump (1 = yes, 0 = no)')
# print('\tAverage F1 score:', np.mean(f1_scores))
# print('\tStdDev F1 score:', np.std(f1_scores))
# print('\tMedian F1 score:', np.median(f1_scores))
# print('\tIQR F1 score:', stats.iqr(f1_scores))
# print('\tSkewness F1 score:', stats.skew(f1_scores))
#
#
# # Extra Tree Model 3
# # Separate Y and X variables
# df_label = df.loc[:, 'speedbump']
# df_feature = df.loc[:, ('Speed', 'X', 'Y', 'z_jolt')]
# Y = df_label.as_matrix()
# X = df_feature.as_matrix()
#
#
# # Prepare for cross-validation
# clf = ExtraTreesClassifier(random_state=0)  # create a MLPClassifier
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
#     # calculate the F1 score
#     f1 = metrics.f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
#     f1_scores.append(f1)
#
#     # calculate the confusion matrix
#     matrix = metrics.confusion_matrix(test_Y, predicted_Y)
#
#
# # Calculate cross-validation average
# print('\n-----------------------------------')
# print('sklearn_Models.ensemble.ExtraTreeClassifier Model 3')
# print('\tFeatures: speed, X-accel, Y-accel, Z-jolt')
# print('\tLabels: speedbump (1 = yes, 0 = no)')
# print('\tAverage F1 score:', np.mean(f1_scores))
# print('\tStdDev F1 score:', np.std(f1_scores))
# print('\tMedian F1 score:', np.median(f1_scores))
# print('\tIQR F1 score:', stats.iqr(f1_scores))
# print('\tSkewness F1 score:', stats.skew(f1_scores))
#
#
# # Extra Tree Model 4
# # Separate Y and X variables
# df_label = df.loc[:, 'speedbump']
# df_feature = df.loc[:, ('X', 'Y', 'Z', 'z_jolt')]
# Y = df_label.as_matrix()
# X = df_feature.as_matrix()
#
#
# # Prepare for cross-validation
# clf = ExtraTreesClassifier(random_state=0)  # create a MLPClassifier
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
#     # calculate the F1 score
#     f1 = metrics.f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
#     f1_scores.append(f1)
#
#     # calculate the confusion matrix
#     matrix = metrics.confusion_matrix(test_Y, predicted_Y)
#
#
# # Calculate cross-validation average
# print('\n-----------------------------------')
# print('sklearn_Models.ensemble.ExtraTreeClassifier Model 4')
# print('\tFeatures: X-accel, Y-accel, Z-accel, Z-jolt')
# print('\tLabels: speedbump (1 = yes, 0 = no)')
# print('\tAverage F1 score:', np.mean(f1_scores))
# print('\tStdDev F1 score:', np.std(f1_scores))
# print('\tMedian F1 score:', np.median(f1_scores))
# print('\tIQR F1 score:', stats.iqr(f1_scores))
# print('\tSkewness F1 score:', stats.skew(f1_scores))
