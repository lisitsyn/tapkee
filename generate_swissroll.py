import numpy
N = 1000
tt = numpy.array((3*numpy.pi/2)*(1+2*numpy.random.rand(N)))
height = numpy.array((numpy.random.rand(N)-0.5))
X = numpy.array([tt*numpy.cos(tt), 10*height, tt*numpy.sin(tt)])

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[0],X[1],X[2],c=tt,cmap=cm.Spectral)
show()

numpy.savetxt('colormap.dat',tt)
numpy.savetxt('input.dat',X.T)
