import csv
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
import Orange
from random import shuffle
from sklearn import cross_validation, linear_model
import sklearn
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
    return crit_func + regularization                   # crit_func + regularization


def criteria_function_grad( theta, X_samples, y, lambda_ ):
    """
    Gradient of critera function
    :param theta:           Parameters of function 'func'
    :param args:
    :return:                Gradient of a function at particular spot
    """

    if ( theta == None ):                               # support some of my laziness
        length = X_samples[0].shape[0]
        theta = np.ones( length + 1 )

    n = X_samples.shape[0]                                # number of samples
    return ( 1/float(n) ) * (X_samples.dot(theta) - y).dot(X_samples) + lambda_ *2 * theta  # 1/n sum( ( f(x_ij) - y ) * x_i)


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


def check_grad(X_samples, y):
    lambda_= 0.225
    # check gradient
    theta = np.ones( X_samples.shape[1] )
    ag = criteria_function_grad(theta, X_samples, y, lambda_)
    ng = numerical_grad(lambda params: criteria_function(params, X_samples, y, lambda_), theta, 1e-4)
    print(np.sum((ag - ng)**2))


###############################################################################
# DATA MANAGEMENT
###############################################################################


def stringTOint(matrix):
    matrix1 = []
    for pos in range(len(matrix)):
        i = matrix[pos]
        #i[i == ''] = 0.0
        i = i.astype(np.float)
        matrix1.append(i)
    return np.array( matrix1 )


# LAST COLUMN IS DELETED!!! WARNING
def get_data(filename):
    data = list(csv.reader(open(filename, 'r'), delimiter='\t'))
    #data = Orange.data.Table(filename)

    # ELIMINATE HEADER
    data = np.array( data[3:] )
    return data


def get_best_attributes(n, data, y_compare):
    num_of_attributes = min(n, data.shape[1])
    attribute_positions = list(range(num_of_attributes))
    attribute_positions.sort(key=lambda position:
                                abs((scipy.stats.pearsonr(data[:,position],
                                                         y_compare))
                                [0]), reverse=True)
    return attribute_positions[:n]              # return best n attributes (their positions)


def LinearRegression(X_samples, y_values, lambda_=0.1):
    X_ones_samples = np.column_stack(( np.ones( X_samples.shape[0] ), X_samples ))
    func = lambda theta: \
        criteria_function( theta, X_ones_samples, y_values, lambda_ )
    theta0 = np.ones( X_ones_samples.shape[1] )
    grad = lambda theta: \
        criteria_function_grad( theta, X_ones_samples, y_values, lambda_ )

    theta, _, _ = scipy.optimize.fmin_l_bfgs_b( func, theta0, grad)
    return theta


def predict( theta, X_samples ):
    X_ones_samples = np.column_stack(( np.ones( X_samples.shape[0] ), X_samples ))
    return X_ones_samples.dot(theta)


###############################################################################
# OTHER
###############################################################################

def kfoldcv(X, y):
    k = 10
    kf = cross_validation.KFold(len(y), n_folds=5)
    rmse = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scores = sorted([(abs(scipy.stats.pearsonr(X_train[:, i], y_train)[0]), i)
                         for i in range(X_train.shape[1])], reverse=True)
        X_sel_tr = X_train[:, [i for _, i in scores[:k]]]
        X_sel_tst = X_test[:, [i for _, i in scores[:k]]]

        lrn = linear_model.Ridge(alpha=0.01, fit_intercept=True, solver="lsqr").fit(X_sel_tr, y_train)
        pred = lrn.predict(X_sel_tst)
        rmse.append(np.sqrt(sum((pred - y_test)**2)/len(y_test)))
    return np.mean(rmse)


def run():
    # train_data = Orange.data.Table("../data/train.tab")
    # test_data = Orange.data.Table("../data/test.tab")
    train_data = get_data("../data/train.tab")
    print train_data.shape[1]
    test_data = get_data("../data/test.tab")

    X_samples = (stringTOint( train_data[:,:-2] ))
    X_samples_test = (stringTOint( test_data[:,:-2] ))
    y = (stringTOint( train_data[:, -2] ))

    num_of_attributes = 50
    best_att_positions = get_best_attributes(num_of_attributes, X_samples, y)
    theta = LinearRegression(X_samples[:,best_att_positions], y, lambda_=0.225)
    # print theta
    results = predict( theta, X_samples_test[:, best_att_positions] )

    # f = open('../results/r0.txt', 'w')
    # for i in results:
    #     f.write(str(i) + "\n")

    print kfoldcv(X_samples, y)
run()