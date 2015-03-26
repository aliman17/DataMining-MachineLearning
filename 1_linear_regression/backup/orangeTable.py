import csv
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
import Orange

__author__ = 'Ales'


###############################################################################
# FUNCTION, CRITERIA FUNCTION, GRAD, NUMERICAL GRAD
###############################################################################


def func(theta, x):
    return x.dot(theta)                                 # f = a0 + a1 x1 + a2 x2 + a3 x3 + ...


def criteria_function( theta, X_samples, y_value, lambda_ ):
    """
    Criteria function for function 'func'
    :param theta:       Parameters of function 'func'
    :param args:        #TODO
    :return:            Computed value of a criteria function
    """
    n = X_samples.shape[0]                                # number of samples
    crit_func = ( 1 / float(2*n) ) \
                * sum([ (func(theta, X_samples[i]) - y_value[i])**2
                        for i in range(X_samples.shape[0]) ])
    #TODO regularization
    regularization = lambda_ * sum( theta**2 )          # regularization
    return crit_func #+ regularization                   # crit_func + regularization


def criteria_function_grad( theta, samples, y, lambda_ ):
    """
    Gradient of critera function
    :param theta:           Parameters of function 'func'
    :param args:
    :return:                Gradient of a function at particular spot
    """

    if ( theta == None ):                               # support some of my laziness
        length = samples[0].shape[0]
        theta = np.ones( length + 1 )

    n = samples.shape[0]                                # number of samples
    X = np.hstack((samples, 1))                         # add 1 for afine translation of the line
    return ( 1/float(n) ) * (X.dot(theta) - y).dot(X)   # 1/n sum( ( f(x_ij) - y ) * x_i)


def numerical_grad(f, params, epsilon):
    "Method of finite differences, sanity check to see if our Jgrad is implemented correctly"
    # ag = Jgrad(theta, selected.X, selected.Y)
    # ng = numerical_grad(lambda params: J(params, selected.X, selected.Y), theta, 1e-7)
    # ag, ng # should be about the same
    num_grad = np.zeros_like(params)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad


###############################################################################
# DATA MANAGEMENT
###############################################################################

def get_best_attributes(n, data):
    #n = min(n, data.shape[1])
    num_of_attributes = len( data.domain.attributes )
    attribute_positions = list(range(num_of_attributes))
    attribute_positions.sort(key=lambda position:
                                abs(scipy.stats.pearsonr(data[:, position],
                                                         data[:, data.domain.class_var])
                                [0][0]), reverse=True)
    return attribute_positions[:n]              # return best n attributes (their positions)


def LinearRegression(X_samples, y_values, lambda_=0.1):
    X_ones_samples = np.column_stack(( np.ones( X_samples.shape[0] ), X_samples ))
    func = lambda theta: \
        criteria_function( theta, X_samples, y_values, lambda_ )
    theta0 = np.ones( X_samples.shape[1] )
    grad = lambda theta: \
        criteria_function_grad( theta, X_samples, y_values, lambda_ )

    theta, _, _ = scipy.optimize.fmin_l_bfgs_b( func, theta0, grad)
    return theta


def predict( theta, X_samples ):
    X_ones_samples = np.column_stack(( np.ones( X_samples.shape[0] ), X_samples ))
    return X_ones_samples.dot(theta)





def run():
    train_data = Orange.data.Table("../data/train.tab")
    test_data = Orange.data.Table("../data/test.tab")

    num_of_attributes = 50
    best_att_positions = get_best_attributes(num_of_attributes, train_data)

    X_y = Orange.data.Table(train_data.X[:, best_att_positions], train_data.Y)
    predictor = LinearRegression(X_y.X, X_y.Y, lambda_=0.225)

run()__author__ = 'Ales'
