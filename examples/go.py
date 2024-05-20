#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import subprocess
import re
import tempfile

import numpy as np
from utils import generate_data, plot

def generate_data(type, N=1000, random_state=None):
	rng = np.random.RandomState(random_state)
	if type=='swissroll':
		tt = np.array((3*np.pi/2)*(1+2*rng.rand(N)))
		height = np.array((rng.rand(N)-0.5))
		X = np.array([tt*np.cos(tt), 10*height, tt*np.sin(tt)])
		return X, tt
	if type=='scurve':
		tt = np.array((3*np.pi*(rng.rand(N)-0.5)))
		height = np.array((rng.rand(N)-0.5))
		X = np.array([np.sin(tt), 10*height, np.sign(tt)*(np.cos(tt)-1)])
		return X, tt
	if type=='helix':
		tt = np.linspace(1,N,N).T / N
		tt = tt*2*np.pi
		X = np.r_[[(2+np.cos(8*tt))*np.cos(tt)],
			[(2+np.cos(8*tt))*np.sin(tt)],
			[np.sin(8*tt)]]
		return X, tt
	if type=='twinpeaks':
		X = rng.uniform(-1, 1, size=(N, 2))
		tt = np.sin(np.pi * X[:, 0]) * np.tanh(X[:, 1])
		tt += 0.1 * rng.normal(size=tt.shape)
		X = np.vstack([X.T, tt])
		return X, tt
	if type=='klein':
		u = rng.uniform(0, 2 * np.pi, N)
		v = rng.uniform(0, 2 * np.pi, N)
		x = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.cos(u)
		y = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.sin(u)
		z = np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)

		noise = 0.01
		x += noise * rng.normal(size=x.shape)
		y += noise * rng.normal(size=y.shape)
		z += noise * rng.normal(size=z.shape)
		return np.vstack((x, y, z)), u

	raise Exception('Dataset is not supported')

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

def plot(data, embedded_data, colors='m', method=None):
	fig = plt.figure()
	fig.set_facecolor('white')

	ax_original = fig.add_subplot(121, projection='3d')
	scatter_original = ax_original.scatter(data[0], data[1], data[2], c=colors, cmap=plt.cm.Spectral, s=5, picker=True)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Original', fontsize=9)

	ax_embedding = fig.add_subplot(122)
	scatter_embedding = ax_embedding.scatter(embedded_data[0], embedded_data[1], c=colors, cmap=plt.cm.Spectral, s=5, picker=True)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Embedding' + (' with ' + method) if method else '', fontsize=9, wrap=True)

	highlighted_points = []  # To store highlighted points

	# Function to highlight points on both plots
	def highlight(index):
		# Reset previous highlighted points
		for point in highlighted_points:
			point.remove()
		highlighted_points.clear()

		# Highlight the current point on both scatter plots
		point1 = ax_original.scatter([data[0][index]], [data[1][index]], [data[2][index]], color='white', s=25, edgecolor='black', zorder=3)
		point2 = ax_embedding.scatter([embedded_data[0][index]], [embedded_data[1][index]], color='white', s=25, edgecolor='black', zorder=3)
		highlighted_points.append(point1)
		highlighted_points.append(point2)
		fig.canvas.draw_idle()

	# Event handler for mouse motion
	def on_hover(event):
		if event.inaxes == ax_original:
			cont, ind = scatter_original.contains(event)
		elif event.inaxes == ax_embedding:
			cont, ind = scatter_embedding.contains(event)
		else:
			return

		if cont:
			index = ind['ind'][0]
			highlight(index)

	fig.canvas.mpl_connect('motion_notify_event', on_hover)

	plt.show()

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
