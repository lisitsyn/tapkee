import numpy, datetime, json, subprocess, sys, os, glob
import scipy.misc

def load(dir):
	images = []
	vecs = []
	for f in glob.glob(os.path.join(dir,'*.pgm')):
		image = numpy.array(scipy.misc.imread(f))
		images.append((f,image))
		vecs.append(image.ravel())
	return numpy.vstack(vecs), images

def embed(feature_matrix):
	input_file = 'tmp_cbcl_input'
	numpy.savetxt(input_file,feature_matrix)
	output_file = 'tmp_cbcl_output.dat'
	run_string = './bin/tapkee_cli -i %s -o %s -m ltsa -k 20 --transpose --verbose --benchmark' % (input_file,output_file)
	output = subprocess.check_output(run_string, shell=True)
	embedding = numpy.loadtxt(output_file)
	os.remove(output_file)
	return embedding

def export_json(outfile,embedding,images):
	json_dict = {}
	N = embedding.shape[1]
	print 'N', N
	import scipy.misc
	json_dict['data'] = [{'cx':embedding[0,i], 'cy':embedding[1,i], 'fname':images[i][0]} for i in xrange(N)]
	json.dump(json_dict, open(outfile, 'w'))

def plot_embedding(embedding,images):
	import matplotlib.pyplot as plt
	from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(embedding[0],embedding[1],alpha=0.0)
	for i in xrange(embedding.shape[1]):
		img = numpy.zeros((images[i][1].shape[0],images[i][1].shape[1],4))
		img[:,:,0] = 255*images[i][1]
		img[:,:,1] = 255*images[i][1]
		img[:,:,2] = 255*images[i][1]
		img[:,:,3] = 1
		img[(images[i][1]==28),3] = 0
		imagebox = OffsetImage(img,cmap=plt.cm.gray,zoom=0.2)
		ab = AnnotationBbox(imagebox, (embedding[0][i], embedding[1,i]),pad=0.001,frameon=False)
		ax.add_artist(ab)
	plt.show()

if __name__ == "__main__":
	feature_matrix, images = load('data/cbcl')
	embedding = embed(feature_matrix)
	if len(sys.argv)==3:
		export_json(sys.argv[2],embedding, images)
	plot_embedding(embedding,images)
