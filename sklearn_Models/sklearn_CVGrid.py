# Dependencies
import random
import pandas as pd
import numpy as np
import warnings
from time import time
from scipy import stats
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


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


# Separate Y and X variables
df_label = df.loc[:, 'speedbump']
df_feature = df.loc[:, ('Speed', 'X', 'Y', 'Z', 'z_jolt')]
Y = df_label.as_matrix()
X = df_feature.as_matrix()


# Create a DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)


# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Hyper-parameters: {0}".format(results['params'][candidate]))
            print("")


# Specify parameters and distributions to sample from
param_grid = {"criterion": ["gini", "entropy"],
              "splitter": ["best", "random"],
              "max_depth": [5, 4, 3, 2, 1, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "max_features": ["auto", "log2", None]}


# Run randomized search
grid_search = GridSearchCV(clf, param_grid=param_grid, scoring='f1')
start = time()
grid_search.fit(X, Y)
print("")
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
print("")
report(grid_search.cv_results_)
