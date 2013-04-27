import numpy, datetime, json, subprocess
from modshogun import MultidimensionalScaling, RealFeatures

def embed(fname='mnist2000.dat'):
	run_string = './../../bin/tapkee_cli -i %s -o tmp.dat -m t-sne --verbose --benchmark' % (fname)
	output = subprocess.check_output(run_string, shell=True)
	print output
	return numpy.loadtxt('tmp.dat'), None

def plot_embedding(embedding):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(embedding[0],embedding[1])
	plt.show()

def export_json(embedding):
	json_dict = {}
	N = embedding.shape[1]
	json_dict['data'] = [{'cx':embedding[0,i], 'cy':embedding[1,i]} for i in xrange(N)]
	json.dump(json_dict, open('mnist.json', 'w'))

if __name__ == "__main__":
	embedding, data = embed()
	export_json(embedding)
	plot_embedding(embedding)
