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


# select train ratio for cross-validation
train_ratio = 0.8
data_set = df.values
random.shuffle(data_set)
response_col = 0  # TODO: find the response column
rows = df.shape[0]  # number of rows (data points) in this data frame
train_rows = int(train_ratio * rows) # number of rows in the train set

# the train set (X-variables & Y-variable)
train_X = data_set[1:train_rows, response_col+1 : data_set.shape[1]]  # TODO: verify the independent columns
train_Y = data_set[1:train_rows, response_col]

# the test set (X-variables & Y-variable)
test_X = data_set[train_rows:data_set.shape[0], response_col+1: data_set.shape[1]]   # TODO: verify the independent columns
test_Y = data_set[train_rows:data_set.shape[0], response_col]