import numpy,sys
N = int(sys.argv[1]) 
tt = numpy.array((3*numpy.pi*(numpy.random.rand(N)-0.5)))
height = numpy.array((numpy.random.rand(N)-0.5))
X = numpy.array([numpy.sin(tt), 10*height, numpy.sign(tt)*(numpy.cos(tt)-1)])

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
fig = figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[0],X[1],X[2],c=tt,cmap=cm.Spectral)
show()

numpy.savetxt('colormap.dat',tt)
numpy.savetxt('input.dat',X)
