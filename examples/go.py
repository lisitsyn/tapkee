#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import subprocess

import numpy as np
from utils import generate_data, plot

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
