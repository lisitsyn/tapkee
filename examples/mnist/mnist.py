import numpy, datetime, json
from modshogun import Isomap, RealFeatures

def embed(fname='mnist2000.dat'):
	print '%s reading %s' % (datetime.datetime.now(), fname)
	data = numpy.loadtxt(fname)
	data = data.T
	print '%s there are %d vectors with %d elements in %s' \
			% (datetime.datetime.now(), data.shape[1], data.shape[0], fname)
	features = RealFeatures(data.T)
	converter = Isomap()
	converter.set_k(100)
	converter.set_target_dim(2)
	return converter.apply(features).get_feature_matrix(), data

def plot_embedding(embedding,gc_content=None):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(embedding[0],embedding[1])
	if gc_content:
		fig.colorbar(gc_content,orientation='horizontal')
	plt.show()

def export_json(embedding):
	json_dict = {}
	N = embedding.shape[1]
	json_dict['data'] = [{'embedx':embedding[0,i], 'embedy':embedding[1,i]} for i in xrange(N)]
	json.dump(json_dict, open('mnist.json', 'w'))

if __name__ == "__main__":
	embedding, data = embed()
	export_json(embedding)
	plot_embedding(embedding)
