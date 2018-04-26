from __future__ import print_function
import numpy, datetime, json, subprocess, sys, os

def embed(fname):
	input_file = 'tmp_mnist_input.dat'
	with open(fname,'r') as file:
		json_data = json.load(file)
	N, d = json_data['N'], json_data['d']
	X = numpy.zeros((N,d))
	for i in range(N):
		X[i,json_data['data'][str(i)]] = 1.0
	numpy.savetxt(input_file, X.T, fmt='%.1f', delimiter=',')
	output_file = 'tmp_mnist_output.dat'
	run_string = './bin/tapkee -i %s -o %s -m t-sne --verbose --benchmark' % (input_file,output_file)
	output = subprocess.check_output(run_string, shell=True)
	embedding = numpy.loadtxt(output_file, delimiter=',').T
	os.remove(input_file)
	os.remove(output_file)
	return embedding, X

def plot_embedding(embedding,data):
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(embedding[0],embedding[1],alpha=0.0)
	for i in range(embedding.shape[1]):
		img = numpy.zeros((28,28,4))
		img[:,:,0] = 255*data[i,:].reshape(28,28)
		img[:,:,1] = 255*data[i,:].reshape(28,28)
		img[:,:,2] = 255*data[i,:].reshape(28,28)
		img[:,:,3] = data[i,:].reshape(28,28)
		imagebox = OffsetImage(img,cmap=plt.cm.gray,zoom=0.5)
		ab = AnnotationBbox(imagebox, (embedding[0][i], embedding[1,i]),pad=0.001,frameon=False)
		ax.add_artist(ab)
	plt.title('MNIST embedding')
	plt.axis('off')
	#plt.grid()
	plt.show()

def export_json(outfile,embedding,data):
	json_dict = {}
	N = embedding.shape[1]
	print('N', N)
	import scipy.misc
	for i in range(N):
		img = numpy.zeros((28,28,4))
		img[:,:,0] = 255*data[i,:].reshape(28,28)
		img[:,:,1] = 255*data[i,:].reshape(28,28)
		img[:,:,2] = 255*data[i,:].reshape(28,28)
		img[:,:,3] = 255*data[i,:].reshape(28,28)
		scipy.misc.imsave('mnist/%d.png' % i,img)
	json_dict['data'] = [{'cx':embedding[0,i], 'cy':embedding[1,i], 'fname':'%d.png' % i} for i in range(N)]
	json.dump(json_dict, open(outfile, 'w'))

if __name__ == "__main__":
	embedding, data = embed(sys.argv[1])
	if len(sys.argv)==3:
		export_json(sys.argv[2],embedding,data)
	plot_embedding(embedding,data)
