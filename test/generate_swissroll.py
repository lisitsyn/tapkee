import numpy
import sys

N = int(sys.argv[1])
tt = numpy.array((3*numpy.pi/2)*(1+2*numpy.random.rand(N)))
height = numpy.array((numpy.random.rand(N) - 0.5))
X = numpy.array([tt*numpy.cos(tt), 10*height, tt*numpy.sin(tt)])

numpy.savetxt('colormap.dat', tt, '%.5f', delimiter=',')
numpy.savetxt('input.dat', X.T, '%.5f', delimiter=',')
