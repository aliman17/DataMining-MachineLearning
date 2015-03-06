__author__ = 'Ales'


import csv
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
import random

NUM_OF_ATTRIBUTES = 10

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


def func(sample, koeficients):
    # SUPPORT MY LAZINESS :D
    if ( koeficients == None ):
        length = sample.shape[0]
        koeficients = np.ones( length + 1 )

    # ADD 1 TO THE END, SO THAT WE CAN USE SCALAR PRODUCT TO CREATE
    # f = a0 + a1x1 + a2x2 + a3x3 + ...
    setOFattributes = np.hstack((sample, 1))
    return setOFattributes.dot(koeficients)


# print func (np.array([1,2,3]), np.array([1,1,1,1]))


def criteria_function(koeficients, *args):
    # VSOTA RAZLIK KVADRATOV
    samples, intensity = args[0], args[1]
    return sum([ (func(samples[i], koeficients) - intensity[i])**2
                 for i in range(samples.shape[0]) ])


def derive_crit_function(koeficients, *args):
    samples, intensity = args[0], args[1]

    # CREATE LIST TO STORE ALL DERIVES FOR EACH KOEFICIENT
    Phi = []

    # SUPPORT SOME OF MY LAZYNESS
    if ( koeficients == None ):
        length = samples[0].shape[0]
        koeficients = np.ones( length + 1 )

    # DERIVE PHI, WHICH IS: 2* sum (f(x_ij) - y_i) * x_i

    # COMPUTE J-TH COMPONENT OF PHI WHICH IS A DERIVATE OF J-TH KOEFICIENT
    for j in xrange(koeficients.shape[0]):
        sum = 0

        # ITERATE THROUGH ALL SAMPLES
        for i in xrange(samples.shape[0]):

            # LITTLE REPAIR IF WE ARE AT THE LAST KOEFICIENT WHICH HAS NO ATRIBUTE
            atr_val = 1 if (j == koeficients.shape[0] - 1) else samples[i][j]
            sum += (func(samples[i], koeficients) - intensity[i]) * atr_val

        # STORE THE DERIVATE TO THE RIGHT PLACE
        Phi.append( 2 * sum )
    return Phi

samples = np.array(bestAttributes).T
# print criteria_function (None, samples, intensity)
# print derive_crit_function(None, samples, intensity)


def check_derivation():

    # CREATE RANDOM DATA
    x = np.array([[random.randint(1, 10) for i in range(10)]])
    y = np.array([20])

    length = samples[0].shape[0] + 1
    koeficients = np.ones( length )

    # SET EPSILON
    e = 0.1

    koeficients1 = np.array([1+e]+ ([1] * (length - 1)))
    koeficients2 = np.array([1-e]+ ([1] * (length - 1)))

    d1 = derive_crit_function(koeficients, samples, intensity)
    print "computed derivate", d1

    # KONCNE DIFERENCE
    d2 = np.array([( criteria_function(koeficients1, samples, intensity)
                     - criteria_function(koeficients2, samples, intensity ))
                   / float( 2*e )])
    print "numeric derivate", d2


# check_derivation()


def optimization(samples, intensity):
    # WE HAVE TO MINIMAZE   sum(f - yi)**2, WHERE KOEFICIENTS OF
    # CRITERIA FUNCTIONS MUST BE FOUND

    length = samples[0].shape[0] + 1
    koeficients = np.ones( length )

    #TODO fprime not working jet
    min, _,_ = scipy.optimize.fmin_l_bfgs_b( criteria_function,
                                             koeficients,
                                             fprime=derive_crit_function,
                                             args=(samples,intensity),
                                             approx_grad=True)
    return min


theta = optimization(samples, intensity)[::-1]
print "bfgs", theta


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


dataT = get_data('../data/test.tab')
dataT = [dataT[p] for p in positions]
dataT = stringTOint(dataT)
samples = dataT.T

print "zapisujem"

y = []
f = open('../results/y5.txt', 'w')
for i in samples:
    y = func(i, theta)
    f.write(str(y) + "\n")
