#!/usr/bin/env python
from __future__ import print_function

import numpy
import sys
import os
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

supported_methods = ('lle','ltsa','isomap','mds','pca','kpca','tsne')

def generate_data(type, N=1000):
	if type=='swissroll':
		tt = numpy.array((3*numpy.pi/2)*(1+2*numpy.random.rand(N)))
		height = numpy.array((numpy.random.rand(N)-0.5))
		X = numpy.array([tt*numpy.cos(tt), 10*height, tt*numpy.sin(tt)])
		return X,tt
	if type=='scurve':
		tt = numpy.array((3*numpy.pi*(numpy.random.rand(N)-0.5)))
		height = numpy.array((numpy.random.rand(N)-0.5))
		X = numpy.array([numpy.sin(tt), 10*height, numpy.sign(tt)*(numpy.cos(tt)-1)])
		return X,tt
	if type=='helix':
		tt = numpy.linspace(1,N,N).T / N
		tt = tt*2*numpy.pi
		X = numpy.r_[[(2+numpy.cos(8*tt))*numpy.cos(tt)],
		             [(2+numpy.cos(8*tt))*numpy.sin(tt)],
		             [numpy.sin(8*tt)]]
		return X,tt

	raise Exception('Dataset is not supported')


def embed(data,method):
	if method not in supported_methods:
		raise Exception('Method is not supported by this script')

	input_file = 'tapkee_input_data'
	output_file = 'tapkee_output_data'
	numpy.savetxt(input_file, data.T,delimiter=',')
	tapkee_binary = 'bin/tapkee'
	runner_string = '%s -i %s -o %s -m %s -k 20 --verbose --transpose-output --benchmark' % (tapkee_binary,input_file,output_file,method)
	print('-- To reproduce this use the following command', runner_string)
	output = subprocess.check_output(runner_string,shell=True)
	embedded_data = numpy.loadtxt(output_file,delimiter=',')
	os.remove(input_file)
	os.remove(output_file)
	return embedded_data

def plot(data,embedded_data,colors='m'):
	fig = plt.figure()
	fig.set_facecolor('white')
	ax = fig.add_subplot(121,projection='3d')
	ax.scatter(data[0],data[1],data[2],c=colors,cmap=plt.cm.Spectral)
	plt.axis('tight'); plt.axis('off')
	ax = fig.add_subplot(122)
	ax.scatter(embedded_data[0],embedded_data[1],c=colors,cmap=plt.cm.Spectral)
	plt.axis('tight'); plt.axis('off')
	plt.show()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Graphical example of dimension reduction with Tapkee.')
	parser.add_argument('dataset',type=str,nargs=1,help='A dataset to embed. One of the following: swissroll, scurve, helix.')
	parser.add_argument('method',type=str,nargs=1,help='A method to use. One of the following %s' % str(supported_methods))
	args = parser.parse_args()

	dataset = args.dataset[0]
	method = args.method[0]
	print('-- Loading %s data' % dataset)
	data, colors = generate_data(dataset)
	print('-- Embedding %s data with %s' % (dataset,method))
	embedded_data = embed(data,method)
	print('-- Plotting embedded data')
	plot(data,embedded_data,colors)
