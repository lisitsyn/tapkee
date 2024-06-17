#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import subprocess
import re
import tempfile

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
	input_file = tempfile.NamedTemporaryFile(prefix='tapkee_input')
	output_file = tempfile.NamedTemporaryFile(prefix='tapkee_output')
	np.savetxt(input_file.name, data.T,delimiter=',')
	tapkee_binary = 'bin/tapkee'

	runner_string = '%s -i %s -o %s -m %s -k 20 --precompute --debug --verbose --transpose-output --benchmark' % (
		tapkee_binary, input_file.name, output_file.name, method
	)
	print('-- To reproduce this use the following command `{}`'.format(runner_string))
	process = subprocess.run(runner_string, shell=True, capture_output=True, text=True)
	print(process.stderr)
	if process.returncode != 0:
		raise Exception('Failed to embed')

	if match := re.search(r'Parameter dimension reduction method = \[([a-zA-Z0-9() ]+)\]', process.stderr):
		used_method = match.group(1)
	else:
		used_method = ''

	embedded_data = np.loadtxt(output_file, delimiter=',')
	return embedded_data, used_method

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='Graphical example of dimension reduction with Tapkee.')
	parser.add_argument('dataset', type=str, nargs=1, help='A dataset to embed. One of the following: %s' % str(['swissroll', 'scurve', 'helix', 'twinpeaks']))
	parser.add_argument('method', type=str, nargs=1, help='A method to use. Any of the methods supported by Tapkee')
	args = parser.parse_args()

	dataset = args.dataset[0]
	method = args.method[0]
	print('-- Loading %s data' % dataset)
	data, colors = generate_data(dataset)
	print('-- Embedding %s data with %s' % (dataset, method))
	embedded_data, used_method = embed(data, method)
	print('-- Plotting embedded data')
	plot(data, embedded_data, colors, used_method)
