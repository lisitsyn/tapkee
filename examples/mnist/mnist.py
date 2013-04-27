import numpy, datetime, json, subprocess

def embed(fname='mnist2000.dat'):
	run_string = './../../bin/tapkee_cli -i %s -o tmp.dat -m t-sne --verbose --benchmark' % (fname)
	output = subprocess.check_output(run_string, shell=True)
	print output
	return numpy.loadtxt('tmp.dat'), numpy.loadtxt(fname)

def plot_embedding(embedding,data):
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(embedding[0],embedding[1],alpha=0.0)
	for i in xrange(embedding.shape[1]):
		img = numpy.zeros((28,28,4))
		img[:,:,0] = 255*data[i,:].reshape(28,28)
		img[:,:,1] = 255*data[i,:].reshape(28,28)
		img[:,:,2] = 255*data[i,:].reshape(28,28)
		img[:,:,3] = data[i,:].reshape(28,28)
		imagebox = OffsetImage(img,cmap=plt.cm.gray,zoom=0.5)
		ab = AnnotationBbox(imagebox, (embedding[0][i], embedding[1,i]),pad=0.001,frameon=False)
		ax.add_artist(ab)
	plt.show()

def export_json(embedding,data):
	json_dict = {}
	N = embedding.shape[1]

	import scipy.misc
	for i in xrange(N):
		img = numpy.zeros((28,28,4))
		img[:,:,0] = 255-255*data[i,:].reshape(28,28)
		img[:,:,1] = 255-255*data[i,:].reshape(28,28)
		img[:,:,2] = 255-255*data[i,:].reshape(28,28)
		img[:,:,3] = 255*data[i,:].reshape(28,28)
		scipy.misc.imsave('mnist/%d.png' % i,img)
	json_dict['data'] = [{'cx':embedding[0,i], 'cy':embedding[1,i], 'image':'%d.png' % i} for i in xrange(N)]
	json.dump(json_dict, open('mnist.json', 'w'))

if __name__ == "__main__":
	embedding, data = embed()
	export_json(embedding,data)
	plot_embedding(embedding,data)
