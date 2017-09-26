# Dependencies
import pandas as pd
from sklearn import tree
from sklearn.metrics import f1_score


# Data import and cleaning
df = pd.read_csv("./speedbumps.csv")  # read data from the .csv file
df = df.loc[:, ('speedbump', 'Speed', 'Z', 'z_jolt')]  # only select relevant columns
keywords = ['yes', 'no']
mapping = [1, 0]
df = df.replace(keywords, mapping)
print(df.head(10))


# Split DataFrame into train and test sets
rows = df.shape[0]
train_ratio = 0.8;
train_rows = int(rows * train_ratio)
train = df.iloc[0: train_rows]
test = df.iloc[train_rows: rows]


# Separate Y and X variables
train_Y = train.loc[:, 'speedbump']
train_X = train.loc[:, ('Speed', 'Z', 'z_jolt')]
test_Y = test.loc[:, 'speedbump']
test_X = test.loc[:, ('Speed', 'Z', 'z_jolt')]


# Convert DataFrame to NumPy-Array
train_Y = train_Y.as_matrix()
train_X = train_X.as_matrix()
test_Y = test_Y.as_matrix()
test_X = test_X.as_matrix()


# Start training
clf = tree.DecisionTreeClassifier()  # create a DecisionTreeClassifier
clf = clf.fit(train_X, train_Y)  # fit the training data


# Start testing
predicted_Y = clf.predict(test_X)  # predict on the testing data
f1 = f1_score(test_Y, predicted_Y, average='binary')  # calculate the F1 score
print('Features: speed, vertical acceleration, vertical jolt')
print('Labels: speedbump (1 = yes, 0 = no)')
print('Accuracy: F1 score =', f1)
