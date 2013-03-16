from numpy import *
from pylab import *

X = loadtxt('output.dat')
tt = loadtxt('colormap.dat')
scatter(X[0],X[1],c=tt,cmap=cm.Spectral)
show()
