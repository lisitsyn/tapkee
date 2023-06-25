#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

supported_methods = {
	'lle': 'Locally Linear Embedding',
	'ltsa': 'Local Tangent Space Alignment',
	'isomap': 'Isomap',
	'mds': 'Multidimensional Scaling',
	'pca': 'Principal Component Analysis',
	'kpca': 'Kernel Principal Component Analysis',
	't-sne': 't-distributed Stochastic Neighborhood Embedding',
	'dm': 'Diffusion Map',
}

def generate_data(type, N=1000):
	if type=='swissroll':
		tt = np.array((3*np.pi/2)*(1+2*np.random.rand(N)))
		height = np.array((np.random.rand(N)-0.5))
		X = np.array([tt*np.cos(tt), 10*height, tt*np.sin(tt)])
		return X,tt
	if type=='scurve':
		tt = np.array((3*np.pi*(np.random.rand(N)-0.5)))
		height = np.array((np.random.rand(N)-0.5))
		X = np.array([np.sin(tt), 10*height, np.sign(tt)*(np.cos(tt)-1)])
		return X,tt
	if type=='helix':
		tt = np.linspace(1,N,N).T / N
		tt = tt*2*np.pi
		X = np.r_[[(2+np.cos(8*tt))*np.cos(tt)],
		             [(2+np.cos(8*tt))*np.sin(tt)],
		             [np.sin(8*tt)]]
		return X,tt

	raise Exception('Dataset is not supported')


def embed(data,method):
	if method not in supported_methods:
		raise Exception('Method is not supported by this script')

	input_file = 'tapkee_input_data'
	output_file = 'tapkee_output_data'
	np.savetxt(input_file, data.T,delimiter=',')
	tapkee_binary = 'bin/tapkee'
	runner_string = '%s -i %s -o %s -m %s -k 20 --precompute --verbose --transpose-output --benchmark' % (tapkee_binary, input_file, output_file, method)
	print('-- To reproduce this use the following command', runner_string)
	output = subprocess.check_output(runner_string, shell=True)
	embedded_data = np.loadtxt(output_file, delimiter=',')
	os.remove(input_file)
	os.remove(output_file)
	return embedded_data

def plot(data, embedded_data, colors='m', method=None):
	fig = plt.figure()
	fig.set_facecolor('white')

	ax = fig.add_subplot(121, projection='3d')
	ax.scatter(data[0], data[1], data[2], c=colors, cmap=plt.cm.Spectral, s=5)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Original', fontsize=9)

	ax = fig.add_subplot(122)
	ax.scatter(embedded_data[0], embedded_data[1], c=colors, cmap=plt.cm.Spectral, s=5)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Embedding' + (' with ' + method) if method else '', fontsize=9)

	plt.show()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Graphical example of dimension reduction with Tapkee.')
	parser.add_argument('dataset', type=str, nargs=1, help='A dataset to embed. One of the following: %s' % str(['swissroll', 'scurve', 'helix']))
	parser.add_argument('method', type=str, nargs=1, help='A method to use. One of the following %s' % str(list(supported_methods.keys())))
	args = parser.parse_args()

	dataset = args.dataset[0]
	method = args.method[0]
	print('-- Loading %s data' % dataset)
	data, colors = generate_data(dataset)
	print('-- Embedding %s data with %s' % (dataset,method))
	embedded_data = embed(data, method)
	print('-- Plotting embedded data')
	plot(data, embedded_data, colors, supported_methods[method])
