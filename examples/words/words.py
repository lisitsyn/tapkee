#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from modshogun import KernelLocallyLinearEmbedding, CustomKernel
import json
import random
import difflib

def word_kernel(words):
	N = len(words)
	dist_matrix = np.zeros([N,N])
	for i in xrange(N):
		for j in xrange(i,N):
			s = difflib.SequenceMatcher(None,words[i],words[j])
			dist_matrix[i,j] = s.ratio()
	dist_matrix = 0.5*(dist_matrix+dist_matrix.T) 
	return CustomKernel(dist_matrix)

def embed(filename='words.dat'):
	print 'loading'
	words = []
	with open(filename) as f:
		words.extend([str.rstrip() for str in f.readlines()])
	print 'loaded', words

	converter = KernelLocallyLinearEmbedding()
	converter.set_k(20)
	converter.set_target_dim(2)
	converter.parallel.set_num_threads(1)
	embedding = converter.embed_kernel(word_kernel(words)).get_feature_matrix()
	return embedding,words

def export_json(embedding, words):
	data = {}
	N = len(words)
	data['data'] = [{'string':words[i],'cx':embedding[0,i], 'cy':embedding[1,i]} for i in xrange(N)]
	json.dump(data, open('words.json', 'w'))

def plot_embedding(embedding):
	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)
	ax.scatter(embedding[0,:],embedding[1,:],alpha=0.4,cmap=plt.cm.Spectral)
	plt.axis('off')
	plt.show()

if __name__ == "__main__":
	embedding, words = embed()
	export_json(embedding,words)
	plot_embedding(embedding)
