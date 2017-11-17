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

df_test = pd.read_csv("./speedbumps_6.csv")
df_test = df_test.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]
df_test = df_test.replace(keywords, mapping)
testY = df_test.loc[:, 'speedbump']
testX = df_test.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]




# Decision Tree Model 1
# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]

numPoints = 3;
feature1 = "X"
feature2 = "Y"
feature3 = "Z"
feature4 = "Speed"
feature5 = "z_jolt"
feature6 = ""

for i in range(1,numPoints+1):
    if feature1 != "":
        new_feature = (feature1, i)
        df_feature[new_feature] = df_feature[feature1].shift(i)
        testX[new_feature] = testX[feature1].shift(i)
    if feature2 != "":
        new_feature = (feature2, i)
        df_feature[new_feature] = df_feature[feature2].shift(i)
        testX[new_feature] = testX[feature2].shift(i)
    if feature3 != "":
        new_feature = (feature3, i)
        df_feature[new_feature] = df_feature[feature3].shift(i)
        testX[new_feature] = testX[feature3].shift(i)
    if feature4 != "":
        new_feature = (feature4, i)
        df_feature[new_feature] = df_feature[feature4].shift(i)
        testX[new_feature] = testX[feature4].shift(i)
    if feature5 != "":
        new_feature = (feature5, i)
        df_feature[new_feature] = df_feature[feature5].shift(i)
        testX[new_feature] = testX[feature5].shift(i)
    if feature6 != "":
        new_feature = (feature6, i)
        df_feature[new_feature] = df_feature[feature6].shift(i)
        testX[new_feature] = testX[feature6].shift(i)

df_feature = df_feature[numPoints:]
df_label = df_label[numPoints:]
testX = testX[numPoints:]
testY = testY[numPoints:]

# print(df_feature)
# print(testX)

df_feature.index = range(len(df_feature))
df_label.index = range(len(df_label))



Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_features=None, max_depth=40, min_samples_leaf= 10,min_samples_split=2, random_state=0, presort=True)
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
    precision = metrics.precision_score(test_Y, predicted_Y, average='binary')
    precision = float(precision)

    # calculate recall
    recall = metrics.recall_score(test_Y, predicted_Y, average='binary')
    recall = float(recall)

    # calculate F1 score
    if (precision + recall) != 0:
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)


# Calculate cross-validation average
print('\n-----------------------------------')
print('sklearn.tree.DecisionTreeClassifier Model 1')
print('\tFeatures: speed, X-accel, Y-accel, Z-accel, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', np.mean(f1_scores))
print('\tStdDev F1 score:', np.std(f1_scores))
print('\tMedian F1 score:', np.median(f1_scores))
print('\tIQR F1 score:', stats.iqr(f1_scores))
print('\tSkewness F1 score:', stats.skew(f1_scores))
print('\tZero F1 score:', f1_scores.count(0.00))

predictions = clf.predict(testX)
ser = pd.Series(predictions)
ser.name = "predictions"
print(ser.value_counts())
print('\n-----------------------------------')
testY.index = range(len(testY))
print(testY.value_counts())
output = pd.concat([testY, ser], axis=1)
output.to_csv(path_or_buf='output.csv')