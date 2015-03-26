__author__ = 'Ales'
import csv
import scipy
import scipy.stats
import numpy as np
from matplotlib import pyplot as plt
import random
import Orange

NUM_OF_ATTRIBUTES = 40

def findMin( best10 ):
    pos = 0
    counter = 0
    val = 2
    for pos1, corr in best10:
        if (abs(corr) < abs(val)):
            val = corr
            pos = counter
        counter += 1
    return pos, val

# GET DATA
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
    data = stringTOint( data )
    return data


def stringTOint(matrix):
    matrix1 = []
    for pos in range(len(matrix)):
        i = matrix[pos]
        i[i == ''] = 0.0
        i = i.astype(np.float)
        matrix1.append(i)
    return np.array( matrix1 )

# SELECT ATTRIBUTES
def get_best10_attributes(dataT):
    # TAKE THE BEST 10 ATTIBUTES (position of the attribute and value of correlation), SO RESERVE SPACE FIRST
    best10 = [[-2,0]] * NUM_OF_ATTRIBUTES
    intensity = dataT[-1]
    # CONVERT INTENSITY FROM STRING TO FLOAT
    # intensity[intensity == ''] = 0.0
    # intensity = intensity.astype(np.float)


    for i in xrange(len(dataT)-2):           # dont take CID and intensity column which are the last two, so -2
        column = dataT[i]
        # COLUMN REPRESENTS ONE ATTRIBUTE
        # STRING TO FLOAT
        # column[column == ''] = 0.0
        # column = column.astype(np.float)
        # CORRELATION
        (correlation, _) = scipy.stats.pearsonr(column, intensity)
        correlation = correlation
        pos, val = findMin( best10 )
        if ( correlation > val ):
            best10[pos] = [i, correlation]

    return best10, intensity


# WE HAVE GOT THE BEST 10
# WE KNOW WHICH COLUMNS TO TAKE

def combine_attributes(dataT, best10):
    # create criteria function
    s = [0] * len(dataT[1])
    for pos, val in best10:
        column = dataT[pos]
        # column =  dataT[pos]
        # column[column == ''] = 0.0
        # column = column.astype(np.float)
        for i in range(len(s)):
            s[i] += column[i] * val

    for i in range(len(s)):
         s[i] = s[i] / float(len(best10))
    return normalize( s )

# NORMALIZE ATTRIBUTE
def normalize( X ):
    # sum together all values
    s = sum( X )
    # each element devide by sum
    X = [ i/float(s) for i in X ]
    # return new normalized vector
    return X

# NAREDIMO ODVOD ZA NASO FUNKCIJO
def gradient_descent(X, y, alpha=0.01, epochs=1000): # alpha naj bo LEARNING RATE in 1000x bomo stopili naprej po korakih
    theta = np.zeros(X.shape[1]).T   # NAREDI VEKTOR Z NICLAMI DOLZINE KOT JE X
    for i in range(epochs):
        theta = theta - alpha * (X.dot(theta) - y).dot(X)   # namesto vsote pri odvodu, lahko naredimo kar skalarni produkt dveh vektorjev, pa pride isto
    return theta


def get_alpha( dataT ):
    data = dataT.T
    al = []
    for alpha in [0.01, .02, .03, .04]:

        sum_value = 0
        for j in range(1):
            data = np.random.permutation(data)
            training = data[:-60]
            training = training.T
            test = data[-60:]
            test = test.T

            best10, intensity = get_best10_attributes(training)
            s = combine_attributes(training, best10)
            x = np.column_stack([np.ones(len(s)), s])
            y = np.array(intensity)
            theta = gradient_descent(x, y, alpha=alpha)
            # draw(x, y, theta)

            _, intensity = get_best10_attributes(test)
            s = combine_attributes(test, best10)
            x = np.column_stack([np.ones(len(s)), s])
            y = []
            for i in x:
                y.append(theta[1] * i[1] + theta[0])

            y = np.array(y)

            sum = 0
            for k in range(len(y)):
                sum = sum + abs(y[k] - intensity[k])

            sum_value += sum
            print alpha, sum
            print
        al.append([sum_value/float(10), alpha])
    return max(al)

# CRITERIA FUNCTION
def func(params, *args):
    x = np.array(args[0])
    y = np.array(args[1])
    m, b = params
    y_model = m*x+b
    error = y-y_model
    return sum(error**2)

def criteria_function(k, n, x, y):
    return sum((k*x + n - y)**2)

# DERIVATIVE OF CRITERIA FUNCTION
def derive_crit_func(k, n, x, y):
    return np.array([2* sum(x), 2* len(x)])

def check_derive():
    x = np.array([random.randint(1, 10) for i in range(10)])
    y = np.array([random.randint(1, 10) for i in range(10)])
    k, n = 2, 5
    e = 0.0001
    d1 = derive_crit_func(k, n, x, y)
    print d1
    # KONCNE DIFERENCE
    d2 = np.array([(criteria_function(k+e, n, x, y) - criteria_function(k-e, n, x, y))/float(2*e) ,
                   (criteria_function(k, n+e, x, y) - criteria_function(k, n-e, x, y))/float(2*e)])
    print d2
    print d1-d2

#check_derive()

def criteria_function_min(x_true, y_true):
    # WE HAVE TO MINIMAZE   sum(f - yi), WHERE f IS OUR FUNCTION TO PREDICT
    initial_values = np.array([1, 1])
    min, _,_ = scipy.optimize.fmin_l_bfgs_b(func, x0=initial_values, args=(x_true,y_true), approx_grad=True)
    return min

def draw(x, y, theta):
    XX = np.array((min(x), max(x)))
    P = np.column_stack((np.ones(len(XX)), XX))
    YY = P.dot(theta)

    plt.plot(x,y, "o");          # NARISI TOCKE
    plt.plot(XX, YY);       # crta med skrajno levo in skrajno desno TOCKO (PO DEFAULT RISE POLIGON)
    plt.show()


def writefile(name, theta):
    f = open(name, 'w')
    for i in theta:
        f.write(str(i) + "\n")



dataT = get_data('../data/train.tab')
print
best10, intensity = get_best10_attributes(dataT)
s = combine_attributes(dataT, best10)
x = np.column_stack([np.ones(len(s)), s])
y = np.array(intensity)
# NOW WE NEED A LINE BETWEEN X AND Y
# theta = gradient_descent(x, y)
theta = criteria_function_min(s, y)[::-1]
print theta
draw(s, y, theta)
# writefile('theta.txt', theta)

#
# #################################################################
# ##################RUN THE ALGORITHM ON UNKNOWN DATA##############
# #################################################################
#
#
# #
# #
# dataT = get_data('../data/test.tab')
# s = combine_attributes(dataT, best10)
# x = np.column_stack([np.ones(len(s)), s])
# y = []
# for i in x:
#     y.append(theta[1] * i[1] + theta[0])
#
# y = np.array(y)
# print y
# draw(x, y, theta)
# writefile('../results/y3.txt', y)

# #################################################################
# ################## LASSO ########################################
# #################################################################

#
# from sklearn.linear_model import Lasso
#
# alpha = 0.1
# lasso = Lasso(alpha=alpha)
# dataT = get_data('../data/train.tab')
# dataT = stringTOint(dataT[:-1])
#
# dataT2 = get_data('../data/test.tab')
# dataT2 = stringTOint(dataT2[:-2])
#
# y_pred_lasso = lasso.fit(dataT.T, dataT[-1]).predict(dataT2.T)