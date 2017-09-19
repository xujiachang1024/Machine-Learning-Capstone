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


