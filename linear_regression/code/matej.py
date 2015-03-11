__author__ = 'Ales'

# http://docs.orange.biolab.si/3/modules/data.storage.html#data-access
import Orange.data
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
import scipy
from random import shuffle

# # Not really used, but here for testing purposes/sanity check
#
# def gradient_descent(X, y, alpha=0.01, epochs=1000):
#     "Returns theta for gradient descent with no regularization applied"
#     theta = np.zeros((X.shape[1])).T
#     for i in range(epochs):
#         theta = theta - alpha * (X.dot(theta) - y).dot(X)
#     return theta
#
# def gradient_descent_reg(X, y, alpha=0.1, lambda_=0.1, epochs=1000):
#     "Returns theta for gradient descent using L2 regularization"
#     m = X.shape[0]
#     theta = np.zeros((X.shape[1])).T
#     for i in range(epochs):
#         theta = theta * (1 - (alpha * lambda_ / m)) - alpha / m * (X.dot(theta) - y).dot(X)
#     return theta

# http://www.holehouse.org/mlclass/07_Regularization.html
def J(theta, X, y, lambda_):
    "The cost function"
    m = X.shape[0]
    # return 0.5 * sum((X.dot(theta) - y) ** 2) # no regularization
    return 1 / (2 * m) * sum((X.dot(theta) - y) ** 2) + lambda_ * sum(theta[1:] ** 2) # using regularization

def Jgrad(theta, X, y, lambda_):
    "Gradient of the cost function"
    m = X.shape[0]
    # return (X.dot(theta) - y).dot(X) # no regularization
    return 1 / m * (X.dot(theta) - y).dot(X) + lambda_ / m * theta # using regularization

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

def select_attributes(n, data):
    "Returns n best attributes (sorted) from data with greatest correlation with the class"
    atts = list(range(len(data.domain.attributes)))
    atts.sort(key=lambda att: abs(scipy.stats.pearsonr(data[:, att], data[:, data.domain.class_var])[0][0]), reverse=True)
    return atts[:n]

def LinearRegression(X, y, lambda_=0.1):
    """Uses linear regression (J and Jgrad we defined earlier) to try and learn
    to predict y based on X.  Returns a procedure which takes new sample x as
    input and returns the prediction."""
    X = np.column_stack((np.ones(X.shape[0]), X))
    theta, _, _ = scipy.optimize.fmin_l_bfgs_b(lambda t: J(t, X, y, lambda_),
            np.zeros((X.shape[1])),
            lambda t: Jgrad(t, X, y, lambda_))
    def predict(new_x):
        x_with_ones = np.column_stack((np.ones(new_x.shape[0]), new_x))
        return x_with_ones.dot(theta)
    return predict
# TODO: lasso (clanek jss): min (R_a((\Beta_0, \beta_) = min \frac{1}{2N} \sum_{i=1}^{N}(y_i - \beta_0 - x_i^T \cdot \beta)^2 + \delta||\beta||_1
#                        ||\beta||_1 = \sum_{i} |\beta_i|
#                        (5, jss) - poleg L1 norme upostevas se L2 normo -- \beta == \theta
#                        ne mores uporabiti min_l_bfgs, treba na roke

def rand_partition_dataset(k, n):
    "Returns indices that split the list of length n on k partitions"
    indices = list(range(n))
    shuffle(indices)
    partition_size = len(indices)//k
    return [indices[i:i+partition_size] for i in range(0, len(indices), partition_size)]

def RMSE(h, y): return mean_squared_error(h, y)**0.5

def kfoldcv(k, X, Y, learner, **learner_args):
    "Perform k-fold cross validation on data using learner. Returns RMSE."
    partitions = rand_partition_dataset(k, Y.shape[0])
    rmses = []
    for test in partitions:
        train = [inner for outer in partitions for inner in outer if outer != test]
        predictor = learner(X[train], Y[train], **learner_args)
        h = predictor(X[test])
        rmses.append(RMSE(h, Y[test]))
    return np.mean(rmses)

# test with:
# kfoldcv(10, optimized_train_data.X, optimized_train_data.Y, LinearRegression, lambda_=0.2)

def optimize_parameters(train_data, sorted_atts):
    "prints the results of running k-fold cv on training data with various number of top attributes/lambda values"
    glob_min_l = []
    for n_atts in range(45, 65, 1):
        train_selected = Orange.data.Table(train_data.X[:, sorted_atts[:n_atts]], train_data.Y)
        print(n_atts)
        min_l = [0, 10000]
        for l in np.arange(0, 1.0, 0.025):
            kf = kfoldcv(5, train_selected.X, train_selected.Y, LinearRegression, lambda_=l)
            glob_min_l.append((n_atts, l, kf))
    return glob_min_l
# # for optimizing parameters lambda_ and number of best attributes
# results = optimize_parameters(train_data, sorted_atts)
# results.sort(key=lambda r: r[2])
# results[:30]

train_data = Orange.data.Table("../data/train.tab")
test_data = Orange.data.Table("../data/test.tab")

sorted_atts = select_attributes(train_data.X.shape[1], train_data)
n_best_atts = 55

optimized_train_data = Orange.data.Table(train_data.X[:, sorted_atts[:n_best_atts]], train_data.Y)
predictor = LinearRegression(optimized_train_data.X, optimized_train_data.Y, lambda_=0.225)

h = predictor(test_data.X[:, sorted_atts[:n_best_atts]])

def clamp(n, min_val=0, max_val=100):
    "Clamps n between min_val and max_val"
    return min(max_val, max(min_val, n))

def compose(f1, f2):
    "Simple composition of f1 and f2"
    return lambda arg: f1(f2(arg))

outfile = open('predictions.txt', 'w')
outfile.write("\n".join(list(map(compose(str, clamp), h))))