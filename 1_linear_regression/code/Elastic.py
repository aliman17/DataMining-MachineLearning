import csv
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
import Orange
from random import shuffle
from sklearn import cross_validation, linear_model
from sklearn import preprocessing
import sklearn
__author__ = 'Ales'


###############################################################################
# FUNCTION, CRITERIA FUNCTION, GRAD, NUMERICAL GRAD
###############################################################################


def func(theta, x):
    return x.dot(theta)                                 # f = a0 + a1 x1 + a2 x2 + a3 x3 + ...


def criteria_function( theta, X_samples, y_value, lambda_ ):

    n = X_samples.shape[0]                                # number of samples
    crit_func = ( 1 / float(2*n) ) \
                * sum([ (func(theta, X_samples[i]) - y_value[i])**2
                        for i in range(X_samples.shape[0]) ])

    regularization = lambda_ * sum( theta**2 )          # regularization
    return crit_func + regularization                   # crit_func + regularization


def criteria_function_grad( theta, X_samples, y, lambda_ ):
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


def check_grad(X, y):
    lambda_= 0.225
    # check gradient
    theta = np.ones( X.shape[1] )
    ag = criteria_function_grad(theta, X, y, lambda_)
    ng = numerical_grad(lambda params: criteria_function(params, X, y, lambda_), theta, 1e-4)
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


def standardize(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X)


def get_best_attributes(n, data, y_compare):
    num_of_attributes = min(n, data.shape[1])
    attribute_positions = list(range(num_of_attributes))
    attribute_positions.sort(key=lambda position:
                                abs((scipy.stats.pearsonr(data[:,position],
                                                         y_compare))
                                [0]), reverse=True)
    return attribute_positions[:n]              # return best n attributes (their positions)


def LinearRegression(X, y_train, X_predict, lambda_=0.1):

    X = sklearn.preprocessing.scale(X)
    X_predict = sklearn.preprocessing.scale(X_predict)

    X_ones = np.column_stack(( np.ones( X.shape[0] ), X ))
    func = lambda theta: criteria_function( theta, X_ones, y_train, lambda_ )
    theta0 = np.ones( X_ones.shape[1] )
    grad = lambda theta: criteria_function_grad( theta, X_ones, y_train, lambda_ )
    theta, _, _ = scipy.optimize.fmin_l_bfgs_b( func, theta0, grad )      # grad is missing because of lasso and elastic net, otherwise we add it

    X_predict = np.column_stack(( np.ones( X_predict.shape[0] ), X_predict ))
    return X_predict.dot(theta)


def S(z, gamma):
    return np.sign(z) * np.max((np.abs(z) - gamma), 0)

def elastic_net(X, y, X_predict, lambda_=0.1, alpha=0.1, epochs=10):

    X = sklearn.preprocessing.scale(X)
    X_predict = sklearn.preprocessing.scale(X_predict)

    X = np.column_stack(( np.ones( X.shape[0] ), X ))
    N = X.shape[0]
    theta = np.zeros( ( X.shape[1] ) ).T
    for _ in range(epochs):
        # for each field do it seperately
        print "a"
        for j in range( X.shape[1] ):
            y_j = X.dot(theta) - X[:, j] * theta[j]     # remove j=l (look at the article)
            theta[j] = S( (1/float(N)) * ( y-y_j ).dot( X[:, j]) , lambda_ * alpha) / float( 1 + lambda_ * (1 - alpha) )

    X_predict = np.column_stack(( np.ones( X_predict.shape[0] ), X_predict ))
    return X_predict.dot(theta)


###############################################################################
# OTHER
###############################################################################


def lasso(X_train, y_train, X_test, alpha=0.1):
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=alpha)
    y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
    return y_pred_lasso


def elastic(X_train, y_train, X_test, alpha=0.1):
    from sklearn.linear_model import ElasticNet
    enet = ElasticNet(alpha=alpha, l1_ratio=0.7)
    y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
    return  y_pred_enet


def kfoldcv(X, y, k):
    kf = cross_validation.KFold(len(y), n_folds=5)
    rmse = []
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scores = sorted([(abs(scipy.stats.pearsonr(X_train[:, i], y_train)[0]), i)
                         for i in range(X_train.shape[1])], reverse=True)
        X_sel_tr = X_train[:, [i for _, i in scores[:k]]]
        X_sel_tst = X_test[:, [i for _, i in scores[:k]]]

        # pred = LinearRegression(X_sel_tr, y_train, X_sel_tst, lambda_=0.01)
        pred = lasso(X_sel_tr, y_train, X_sel_tst, alpha=2)
        # pred = elastic(X_sel_tr, y_train, X_sel_tst, alpha=0.5)
        # pred = elastic_net(X_sel_tr, y_train, X_sel_tst, lambda_=0.1, alpha=0.5, epochs=100)
        rmse.append(np.sqrt(sum((pred - y_test)**2)/len(y_test)))
    return np.mean(rmse)





def run():
    # train_data = Orange.data.Table("../data/train.tab")
    # test_data = Orange.data.Table("../data/test.tab")
    k = 30
    train_data = get_data("../data/train.tab")
    test_data = get_data("../data/test.tab")

    X = (stringTOint( train_data[:,:-2] ))
    X_predict = (stringTOint( test_data[:,:-2] ))
    y_train = (stringTOint( train_data[:, -2] ))

    # pred = LinearRegression(X, y_train, X_predict, lambda_=0.1)
    pred = elastic_net(X, y_train, X_predict, lambda_=0.5, alpha=0.5, epochs=100)
    # pred = lasso(X_train, y_train, X_test)
    # pred = elastic(X_train, y_train, X_test)

    y = 0
    if (y == 1):
        f = open('../results/e3.txt', 'w')
        for i in pred:
            f.write(str(i) + "\n")

        print "Zapisano"

    # check_grad(X, y_train)
    print pred
run()


