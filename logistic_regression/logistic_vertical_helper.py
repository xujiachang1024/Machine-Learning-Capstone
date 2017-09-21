# code reference:
# https://github.com/hhl60492/newton_logistic/blob/master/main.py


import pandas as pd
import numpy as np
import random
from sklearn import preprocessing


# sigmoid() function: logistic regression curve
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Data import and processing
df = pd.read_csv("./speedbumps.csv")  # read data from the .csv file
df = df.loc[:, ('speedbump', 'Speed', 'Z', 'z_jolt')]  # only select relevant columns
keywords = ['yes', 'no']
mapping = [1, 0];
df = df.replace(keywords,mapping)
print(df.head(10))


# TODO: normalize data frame (I don't understand this part yet)
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

    # row iterator loop
    for j in range(train_X.shape[0]):

        # compute sigmoid outputs
        sig_out[j] = sigmoid(np.dot(train_X[j], params[:, ]))
        diff[j] = sig_out[j] - train_Y[j]

        # compute gradient vector of negative log likelihood
        data[:, j] = train_X[j].transpose()
        gradient[:, 0] = gradient[:, 0] + np.multiply(train_X[j].transpose(), diff[j])

    print("Epoch %d" % i)
    print("Train RMSE %0.4f" % np.sqrt(np.dot(diff[j], diff[j]) / len(diff)))

    # compute Hessian
    sig_der = np.diag(np.multiply(sig_out, np.subtract(1, sig_out)))
    hess = np.matmul(np.matmul(data, sig_der), np.transpose(data))

    # invert Hessian
    hess = np.linalg.inv(hess)

    # do the weight update
    params[:, ] = params[:, ] - step_size * np.matmul(hess, gradient)

    # testing
    sig_out_test = [0] * test_X.shape[0]
    diff_test = [0] * test_X.shape[0]
    for k in range(test_X.shape[0]):
        # compute sigmoid outputs
        sig_out_test[k] = sigmoid(np.dot(test_X[k], params[:, ]))
        diff_test[k] = sig_out[k] - test_Y[k]

    print("Test RMSE %0.4f" % np.sqrt(np.dot(diff_test, diff_test) / len(diff_test)))
