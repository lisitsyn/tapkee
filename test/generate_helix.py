import numpy, sys

N = int(sys.argv[1])
# Generate Helix
tt = numpy.linspace(1,N,N).T / N
tt = tt*2*numpy.pi
X = numpy.r_[ [(2+numpy.cos(8*tt))*numpy.cos(tt)],
              [(2+numpy.cos(8*tt))*numpy.sin(tt)],
              [numpy.sin(8*tt)] ]

import pylab
from mpl_toolkits.mplot3d import Axes3D

# Plot helix
fig = pylab.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[0],X[1],X[2],c=tt,cmap=pylab.cm.Spectral)
pylab.show()

# Save data
numpy.savetxt('colormap.dat', tt)
numpy.savetxt('input.dat', X)
