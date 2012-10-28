from numpy import *
from pylab import *

X = loadtxt('output.dat')
scatter(X[:,0],X[:,1])
show()
