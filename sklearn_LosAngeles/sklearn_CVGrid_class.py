import random
import pandas as pd
import numpy as np
from time import time
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class CVGrid:

    def __init__(self):

        self.__DT_CONST = "DT"
        self.__RF_CONST = "RF"
        self.__GB_CONST = "GB"
        self.__MLP_CONST = "MLP"
        self.__LOG_CONST = "LOG"
        self.__UNDEFINED_CONST = "Undefined"

        self.__df = self.__UNDEFINED_CONST
        self.__Y = self.__UNDEFINED_CONST
        self.__X = self.__UNDEFINED_CONST
        self.__clf = self.__UNDEFINED_CONST
        self.__param_grid = self.__UNDEFINED_CONST
        self.__grid_search = self.__UNDEFINED_CONST

    def read_csv(self, file_name="./los_angeles_10_labeled.csv"):

        self.__df = pd.read_csv(file_name)
        keywords = ['yes', 'no']
        mapping = [1, 0]
        self.__df = self.__df.replace(keywords, mapping)
        self.__Y = self.__df.loc[:, "speedbump"].as_matrix()
        self.__X = self.__df.loc[:, ('Speed', 'vert_accel', 'vert_jolt', 'sq_vert_accel_ratio_speed', 'sq_vert_jolt_ratio_speed')].as_matrix()

    def select_model(self, model="DT"):

        if model == self.__DT_CONST:
            self.__clf = DecisionTreeClassifier(random_state=0)
            self.__param_grid = {"criterion": ["gini", "entropy"],
                                 "splitter": ["best", "random"],
                                 "max_depth": [10, 7, 6, None],
                                 "min_samples_split": [2, 3, 10],
                                 "min_samples_leaf": [1, 3, 10],
                                 "max_features": [3, 4, None]}

        elif model == self.__RF_CONST:
            self.__clf = RandomForestClassifier(random_state=0)
            self.__param_grid = {"criterion": ["gini", "entropy"],
                                 "n_estimators": [9, 10, 11],
                                 "max_depth": [5, 4, 3, 2, 1, None],
                                 "min_samples_split": [2, 3, 10],
                                 "min_samples_leaf": [1, 3, 10],
                                 "max_features": ["auto", "log2", None]}

        elif model == self.__GB_CONST:
            self.__clf = GradientBoostingClassifier(random_state=0)
            self.__param_grid = {"n_estimators": [100, 150, 200],
                                 "max_depth": [10, 5, None],
                                 "min_samples_split": [6, 10],
                                 "min_samples_leaf": [6, 10],
                                 "max_features": [3, 4, None]}

        elif model == self.__MLP_CONST:
            self.__clf = MLPClassifier(random_state=0)
            self.__param_grid = {"solver": ['lbfgs', 'sgd', 'adam'],
                                 "hidden_layer_sizes": [(10, 5), (15, 10), (8, 4), (10, 3)],
                                 'max_iter': [1000, 2000]}

        elif model == self.__LOG_CONST:
            self.__clf = LogisticRegression(penalty='l2')
            self.__param_grid = {"solver": ['newton-cg', 'lbfgs', 'sag'],
                                 "max_iter": [100, 1000, 2000],
                                 'multi_class': ['ovr', 'multinomial']}

        else:
            self.__clf = self.__UNDEFINED_CONST
            self.__param_grid = self.__UNDEFINED_CONST
            print("Alert: Invalid model selection!")

    def run_grid_search(self, scoring="f1"):

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

        self.__grid_search = GridSearchCV(self.__clf, param_grid=self.__param_grid, scoring=scoring)
        start = time()
        self.__grid_search.fit(self.__X, self.__Y)
        print("")
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
              % (time() - start, len(self.__grid_search.cv_results_['params'])))
        print("")
        report(self.__grid_search.cv_results_)


def main():
    
    random.seed(0)
    warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
    cv_grid = CVGrid()
    cv_grid.read_csv()
    cv_grid.select_model()
    cv_grid.run_grid_search()


main()
