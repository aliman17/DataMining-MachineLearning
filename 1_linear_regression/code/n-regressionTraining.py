__author__ = 'Ales'


import csv
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
import Orange

NUM_OF_ATTRIBUTES = 1

def findMin( best10 ):
    pos = 0
    counter = 0
    val = 2
    for pos1, corr, _ in best10:
        if (abs(corr) < abs(val)):
            val = corr
            pos = counter
        counter += 1
    return pos, val


def stringTOint(matrix):
    matrix1 = []
    for pos in range(len(matrix)):
        i = matrix[pos]
        i[i == ''] = 0.0
        i = i.astype(np.float)
        matrix1.append(i)
    return np.array( matrix1 )


# LAST COLUMN IS DELETED!!! WARNING
def get_data(filename):
    data = list(csv.reader(open(filename, 'r'), delimiter='\t'))
    #data = Orange.data.Table(filename)

    # ELIMINATE HEADER
    data = np.array(data[3:])

    # TRANSPOSE, so that list is a column, and we have list of lists
    data = data.T
    data =  data[:-1]

    # STRING TO FLOAT
    return data

dataT = get_data('../data/train.tab')
dataT = stringTOint( dataT )

def get_best10_attributes(dataT):

    # TAKE THE BEST 10 ATTIBUTES (position of the attribute
    # and value of correlation), SO RESERVE SPACE FIRST
    best10 = [[-2,0, -1]] * NUM_OF_ATTRIBUTES
    intensity = dataT[-1]

    # ITERATE THROUGH ALL ATTRIBUTES AND SELECT THE BEST 10 OF THEM
    # dont take CID and intensity column which are the last two, so -2
    for i in xrange(len(dataT)-2):
        column = dataT[i]

        # COMPUTE CORRELATION WHICH IS NEEDET TO DETERMIND HOW GOOD IS THE ATTRIBUTE
        (correlation, _) = scipy.stats.pearsonr(column, intensity)

        # FIND THE WORST ATTRIBUTE WHICH WILL BE REPLACED WITH THE BETTER ONE IF SO
        pos, val = findMin( best10 )
        if ( correlation > val ):
            best10[pos] = [dataT[i], correlation, i]

    # EXTRACT ONLY ATTRIBUTES, NO CORRELATION VALUES NEEDED
    pos = [p for _, _, p in best10]
    best10 = [vec for vec, _,_ in best10]


    # RETURN MATRIX OF THE BEST ATTIBUTES AND INTENSITY ATTRIBUTE
    return best10, intensity, pos

bestAttributes, intensity, positions = get_best10_attributes(dataT)


def normalizeVector( X ):
    s = sum( X )
    return [ i/float(s) for i in X ]


def normalizeMatrix( X ):
    matrix = []
    for i in range(len(X)):
        column = X[i]
        # sum together all values
        s = sum( column )
        # each element devide by sum
        matrix.append ( [ 0 if (s == 0) else i/float(s) for i in column ] )
        # return new normalized matrix
    return matrix


bestAttributes = normalizeMatrix(bestAttributes)
intensity = normalizeVector( intensity )
intensity = np.array(intensity)


def func(X, theta, lambda_):
    if ( theta == None ):                               # support some of my laziness
        length = X.shape[0]
        theta = np.ones( length + 1 )
    X = np.hstack((X, 1))                               # add 1 for afine translation of the line
    return X.dot(theta)                                 # f = a0 + a1 x1 + a2 x2 + a3 x3 + ...


# print func (np.array([1,2,3]), np.array([1,1,1,1]))


def criteria_function( theta, *args ):
    samples, intensity, lambda_ = args[0], args[1], args[2]
    n = samples.shape[0]                                # number of samples
    crit_func = ( 1 / float(2*n) ) \
                * sum([ (func(samples[i], theta) - intensity[i])**2
                        for i in range(samples.shape[0]) ])
    regularization = lambda_ * sum( theta**2 )          # regularization
    return crit_func #+ regularization                   # crit_func + regularization


def criteria_function_grad( theta, *args ):
    samples, y, lambda_ = args[0], args[1], args[2]

    if ( theta == None ):                               # support some of my laziness
        length = samples[0].shape[0]
        theta = np.ones( length + 1 )

    n = samples.shape[0]                                # number of samples
    X = np.hstack((samples, 1))                         # add 1 for afine translation of the line
    return ( 1/float(n) ) * (X.dot(theta) - y).dot(X)   # 1/n sum( ( f(x_ij) - y ) * x_i)


samples = np.array(bestAttributes).T
# print criteria_function (None, samples, intensity)
# print derive_crit_function(None, samples, intensity)


def numerical_grad(f, params, epsilon):
    num_grad = np.zeros_like(theta)
    perturb = np.zeros_like(params)
    for i in range(params.size):
        perturb[i] = epsilon
        j1 = f(params + perturb)
        j2 = f(params - perturb)
        num_grad[i] = (j1 - j2) / (2. * epsilon)
        perturb[i] = 0
    return num_grad

data = Orange.data.Table('iris')
X = data.X
Y = np.eye(3)[data.Y.astype(int)]
theta = np.random.randn(3 * 4)

# compare analytical gradient with gradient obtained by numerical differentiation
ag = criteria_function_grad(theta, samples, intensity)
ng = numerical_grad(lambda params: criteria_function(params, samples, intensity), theta, 1e-4)
print(np.sum((ag - ng)**2))



def optimization(samples, intensity):
    # WE HAVE TO MINIMAZE   sum(f - yi)**2, WHERE KOEFICIENTS OF
    # CRITERIA FUNCTIONS MUST BE FOUND

    length = samples[0].shape[0] + 1
    koeficients = np.ones( length )

    #TODO fprime not working jet
    min, _,_ = scipy.optimize.fmin_l_bfgs_b( criteria_function,
                                             koeficients,
                                             fprime=criteria_function_grad,
                                             args=(samples,intensity),
                                             approx_grad=True)
    return min


# theta = optimization(samples, intensity)[::-1]
# print "bfgs", theta


# JUST FOR COMPARISON WITH BFGS ALGORITHM
def gradient_descent(X, y, alpha=0.01, epochs=1000):
    # alpha naj bo LEARNING RATE in 1000x bomo stopili naprej po korakih
    theta = np.zeros(X.shape[1]).T
    for i in range(epochs):
        theta = theta - alpha * (X.dot(theta) - y).dot(X)
        # namesto vsote pri odvodu, lahko naredimo kar skalarni
        # produkt dveh vektorjev, pa pride isto
    return theta

# print gradient_descent(samples, intensity)


def draw(x, y, theta):
    XX = np.array((min(x), max(x)))
    P = np.column_stack((np.ones(len(XX)), XX))
    YY = P.dot(theta)

    plt.plot(x,y, "o");     # NARISI TOCKE
    plt.plot(XX, YY);       # crta med skrajno levo in skrajno desno TOCKO (PO DEFAULT RISE POLIGON)
    plt.show()


# x = np.column_stack([np.ones(len(bestAttributes[0])), bestAttributes[0]])
# theta = gradient_descent(x, intensity)
# print "grad", theta
# draw(bestAttributes[0], intensity, theta)


# dataT = get_data('../data/test.tab')
# dataT = [dataT[p] for p in positions]
# dataT = stringTOint(dataT)
# samples = dataT.T
#
# print "zapisujem"
#
# y = []
# f = open('../results/y5.txt', 'w')
# for i in samples:
#     y = func(i, theta)
#     f.write(str(y) + "\n")
