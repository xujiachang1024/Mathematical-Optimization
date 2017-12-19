# import dependencies
import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings


# sigmoid function of x
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# set the seed
np.random.seed(0)


# hyper-parameter settings
tolerance = 1e-8  # convergence tolerance
max_iteration = 20  # maximum iterations
lamb = None  # L2-regularization


# data creation settings
r = 0.95  # covariance between x and z
n = 1000  # number of observations
sigma = 1  # variance of noise (how spread out the data is)


# model settings
beta_x, beta_z, beta_v = -4, .9, 1  # true beta coefficients
var_x, var_z, var_v = 1, 1, 4  # variances of inputs


# model specification
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'


# lets keep x and z closely related (height and weight)
x, z = np.random.multivariate_normal([0, 0], [[var_x, r], [r, var_z]], n).T
# blood pressure
v = np.random.normal(0, var_v, n)**3
# create a pandas data frame
A = pd.DataFrame({'x': x, 'z': z, 'v': v})
# compute the log odds for our 3 independent variables
A['log_odds'] = sigmoid(A[['x', 'z', 'v']].dot([beta_x, beta_z, beta_v]) + sigma * np.random.normal(0, 1, n))
# calculate label
A['y'] = [np.random.binomial(1, p) for p in A.log_odds]


# create a data frame that encompasses our input data, model formula, and outputs
y, X = dmatrices(formula, A, return_type='dataframe')
X.head(100)


def catch_singularity(f):

    def silencer(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except np.linalg.LinAlgError:
            warnings.warn('Algorithm terminated - singular Hessian!')
            return args[0]
    return silencer



