import scipy
import scipy.optimize

def func(x):
  return x[0]-x[1], scipy.array(([1.0,-1.0]))

def fprime(x):
   return scipy.array(([1.0,-1.0]))

guess = 1.2, 1.3

#best, val, d = optimize.fmin_l_bfgs_b(func, guess, fprime, 
#approx_grad=True, bounds=bounds, iprint=2)

best, val, d = scipy.optimize.fmin_l_bfgs_b(func, guess, iprint=-1)

print 'Position of the minimum',best, 'and its value',val