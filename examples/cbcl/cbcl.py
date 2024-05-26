import numpy
import datetime
import json
import subprocess
import sys
import os
import glob
import tempfile

import scipy.misc
from PIL import Image

def load(dir):
	images = []
	vecs = []
	for f in glob.glob(os.path.join(dir,'*.pgm')):
		image = numpy.array(Image.open(f))
		images.append((f,image))
		vecs.append(image.ravel())
	return numpy.vstack(vecs), images

def embed(feature_matrix):
	input_file = tempfile.NamedTemporaryFile(prefix='cbcl_input')
	output_file = tempfile.NamedTemporaryFile(prefix='cbcl_output')
	numpy.savetxt(input_file.name, feature_matrix, delimiter=',')
	runner_string = './bin/tapkee -i %s -o %s -m ltsa -k 80 --transpose-output --verbose --benchmark' % (input_file.name, output_file.name)
	process = subprocess.run(runner_string, shell=True, capture_output=True, text=True)
	print(process.stderr)
	if process.returncode != 0:
		raise Exception('Failed to embed')
	embedding = numpy.loadtxt(output_file.name, delimiter=',')
	return embedding

def export_json(outfile, embedding, images):
	json_dict = {}
	N = embedding.shape[1]
	json_dict['data'] = [{'cx':embedding[0,i], 'cy':embedding[1,i], 'fname':images[i][0]} for i in xrange(N)]
	json.dump(json_dict, open(outfile, 'w'))

def plot_embedding(embedding,images):
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(embedding[0],embedding[1],alpha=0.0)
	for i in range(embedding.shape[1]):
		img = numpy.zeros((images[i][1].shape[0], images[i][1].shape[1], 4))
		img[:,:,0] = images[i][1]/255.0
		img[:,:,1] = images[i][1]/255.0
		img[:,:,2] = images[i][1]/255.0
		img[:,:,3] = 1.0
		img[(images[i][1]==28), 3] = 0
		imagebox = OffsetImage(img,cmap=plt.cm.gray,zoom=0.2)
		ab = AnnotationBbox(imagebox, (embedding[0][i], embedding[1,i]), pad=0.001, frameon=False)
		ax.add_artist(ab)
	plt.show()

if __name__ == "__main__":
	feature_matrix, images = load('data/cbcl')
	embedding = embed(feature_matrix)
	if len(sys.argv)==3:
		export_json(sys.argv[2], embedding, images)
	plot_embedding(embedding,images)
