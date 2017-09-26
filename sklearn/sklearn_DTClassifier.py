# Dependencies
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# Set the seed (reproducibility)
random.seed(0)


# Data import and cleaning
df = pd.read_csv("./speedbumps.csv")  # read data from the .csv file
df = df.loc[:, ('speedbump', 'Speed', 'X', 'Y', 'Z', 'z_jolt')]  # only select relevant columns
keywords = ['yes', 'no']
mapping = [1, 0]
df = df.replace(keywords, mapping)
print(df.head(10))


# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('Speed', 'Z', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Prepare for cross-validation
clf = DecisionTreeClassifier()  # create a DecisionTreeClassifier
f1_sum = 0.00  # sum of F1 scores
cv = 10;  # number of cross-validations


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
    f1_sum += f1

    # calculate the confusion matrix
    matrix = metrics.confusion_matrix(test_Y, predicted_Y)

    # print iterative result
    print('\n-----------------------------------')
    print('Iteration ', i)
    print('Features: speed, Z-accel, Z-jolt')
    print('Labels: speedbump (1 = yes, 0 = no)')
    print('F1 score:', f1)
    print(matrix)


# Calculate cross-validation average
f1_average = f1_sum / cv
print('\n-----------------------------------')
print('sklearn Decision Tree Model')
print('\tFeatures: speed, Z-accel, Z-jolt')
print('\tLabels: speedbump (1 = yes, 0 = no)')
print('\tAverage F1 score:', f1_average)
