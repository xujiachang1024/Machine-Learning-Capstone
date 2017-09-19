import pandas as pd
import numpy as np
import math
import random
from sklearn import preprocessing


# sigmoid() function: logistic regression curve
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# TODO: read .csv file
df = pd.read_csv("./data.csv")


# TODO: convert strings to numerical values
keywords = ['yes', 'no', 'HTML', 'Plain', 'none', 'big', 'small']
mapping = [1, 0, 0, 1, 0, 1, 2]
df = df.replace(keywords, mapping)


# TODO: normalize data frame (I don't )
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(np_scaled)


# Select train ratio for cross-validation
train_ratio = 0.8
data_set = df.values
random.shuffle(data_set)
response_col = 0  # TODO: find the response column
rows = df.shape[0]  # number of rows (data points) in this data frame
train_rows = int(train_ratio * rows) # number of rows in the train set

# The train set (X-variables & Y-variable)
train_X = data_set[1: train_rows, response_col+1: data_set.shape[1]]  # TODO: verify the independent columns
train_Y = data_set[1: train_rows, response_col]

# The test set (X-variables & Y-variable)
test_X = data_set[train_rows: data_set.shape[0], response_col+1: data_set.shape[1]]   # TODO: verify the independent columns
test_Y = data_set[train_rows: data_set.shape[0], response_col]


# Hyper-parameter setup
epochs = 1000  # number of iterations
step_size = 0.01  # coefficient change per iteration
params_num = test_X.shape[1]  # number of columns in the test set (X-variables)
params = np.zeros((params_num, 1))  # a column vector with shape (paras_num, 1)
params[:, 0] = np.random.uniform(low=-0.5, high=0.5, size=(params_num,))  # fill params with random [-0.5, 0.5)


# Loop for 1,000 iterations
for i in range(epochs):

    # shuffle the test set (X-variables)
    random.shuffle(test_X)

    sig_out = [0] * train_X.shape[0]  # sigmoid output array (predicted Y)
    diff = [0] * train_X.shape[0] # difference between sigmoid output (predicted Y) and actual Y from the train set
    gradient = np.zeros((train_X.shape[1], 1))  # a column vector with shape (train_rows, 1)
    data = np.zeros((train_X.shape[1], train_X.shape[0]))  # a zero matrix with shape (train_cols, train_rows)
    sig_der = np.zeros((train_X.shape[0], train_X.shape[0]))  # a zero matrix with shape (train_rows, train_rows)

    # iterate through each row in the train set
    for j in range(train_X.shape[0]):

        # compute sigmoid outputs
        sig_out[j] = sigmoid(np.dot(train_X[j], params[:, ]))
        diff[j] = sig_out[j] - train_Y[j]

        # compute gradient vector of negative log likelihood
        data[:, j] = train_X[j].transpose()
        gradient[:, 0] = gradient[:, 0] + np.multiply(train_X[j].transpose(), diff[j])