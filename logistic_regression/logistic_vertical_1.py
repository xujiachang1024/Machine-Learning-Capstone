# Dependencies
import pandas as pd  # data frame
import numpy as np  # matrix math
import patsy  # matrix data structure
import warnings  # error logging


# sigmoid() function: logistic regression model
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Hyper-parameter setup
np.random.seed(0)  # set the seed (reproducibility)
tol = 1e-8  # convergence tolerance
max_iter = 20  # maximum allowed iterations
lam = None  # L2 regularization
formula = 'speedbump ~ Speed + Z + z_jolt'  # TODO: specify the model that we want to fit


# Data import and processing
df = pd.read_csv("./speedbumps.csv")  # read data from the .csv file
df = df.loc[:, ('speedbump', 'Speed', 'Z', 'z_jolt')]  # only select relevant columns
keywords = ['yes', 'no']
mapping = [1, 0];
df = df.replace(keywords, mapping)
print(df.head(10))


# Split the DataFrame into response variable and independent variables
y, X = patsy.dmatrices(formula, df, return_type='dataframe')
print(X.head(10))


# Error checking: to catch singular matrix errors
def catch_singularity(f):
    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated: singular Hessian!')
            return args[0]
    return silencer


# Maximum likelihood estimation: Newton's Method #1
# This function does NOT compute the full inverse of the Hessian
@catch_singularity
def newton_step(curr, X, lam=None):
    # one naive step of Newton's Method

    # how to compute inverse:
    # http://www.mathwarehouse.com/algebra/matrix/images/square-matrix/inverse-matrix.gif

    # compute necessary objects
    # create probability matrix, miniminum 2 dimensions, transpose (flip it)
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    # create weight matrix from it
    W = np.diag((p * (1 - p))[:, 0])
    # derive the hessian
    hessian = X.T.dot(W).dot(X)
    # derive the gradient
    grad = X.T.dot(y - p)

    # L2 regularization step (to avoid over-fitting)
    if lam:
        # Return the least-squares solution to a linear matrix equation
        step, *_ = np.linalg.lstsq(hessian + lam * np.eye(curr.shape[0]), grad)
    else:
        step, *_ = np.linalg.lstsq(hessian, grad)

    # update our beta
    beta = curr + step

    return beta


# Maximum likelihood estimation: Newton's Method #2
# This function does compute the actual inverse of the Hessian
@catch_singularity
def alt_newton_step(curr, X, lam=None):
    # another naive step of Newton's Method

    # compute necessary objects
    p = np.array(sigmoid(X.dot(curr[:, 0])), ndmin=2).T
    W = np.diag((p * (1 - p))[:, 0])
    hessian = X.T.dot(W).dot(X)
    grad = X.T.dot(y - p)

    # L2 regularization step (to avoid over-fitting)
    if lam:
        # Compute the inverse of a matrix.
        step = np.dot(np.linalg.inv(hessian + lam * np.eye(curr.shape[0])), grad)
    else:
        step = np.dot(np.linalg.inv(hessian), grad)

    # update our weights
    beta = curr + step

    return beta


# Convergence setup
def check_coefs_convergence(beta_old, beta_new, tol, iters):
    # checks whether the coefficients have converged in the l-infinity norm.
    # returns True if they have converged, False otherwise.

    # calculate the change in the coefficients
    coef_change = np.abs(beta_old - beta_new)

    # if change hasn't reached the threshold and we have more iterations to go, keep training
    return not (np.any(coef_change > tol) & (iters < max_iter))


# Initial coefficients (weight values), 2 copies, we'll update one TODO: substitute X according to our data
beta_old, beta = np.ones((len(X.columns), 1)), np.zeros((len(X.columns), 1))
iter_count = 0  # number of iterations we've done so far
coefs_converged = False  # have we reached convergence?


# Training step: if we haven't reached convergence...
while not coefs_converged:

    # set the old coefficients to our current
    beta_old = beta

    # perform a single step of newton's optimization on our data, set our updated beta values
    beta = alt_newton_step(beta, X, lam=lam)  # TODO: substitute X according to our data

    # increment the number of iterations
    iter_count += 1

    # check for convergence between our old and new beta values
    coefs_converged = check_coefs_convergence(beta_old, beta, tol, iter_count)


print('Iterations : {}'.format(iter_count))
print('Beta : {}'.format(beta))