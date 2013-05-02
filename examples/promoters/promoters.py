import numpy,os,re,sys,datetime,random,json
from modshogun import WeightedDegreeStringKernel, StringCharFeatures, DNA, MultidimensionalScaling, KernelDistance

def embed(file='data/mml.txt'):
	strings = []
	with open(file) as f:
		strings = [s.rstrip() for s in f.readlines()]
	
	features = StringCharFeatures(strings,DNA)
	kernel = WeightedDegreeStringKernel(10)
	distance = KernelDistance(1.0,kernel)
	distance.init(features,features)
	converter = MultidimensionalScaling()
	converter.set_target_dim(2)
	return converter.embed_distance(distance).get_feature_matrix(), strings

def plot_embedding(data,gc_content=None):
	import matplotlib.pyplot as plt
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.scatter(data[0],data[1])
	plt.title('Promoters embedding')
	plt.grid()
	if gc_content:
		fig.colorbar(gc_content,orientation='horizontal')
	plt.show()

def export_json(outfile, embedding, strings):
	data = {}
	N = len(strings)
	data['data'] = [{'string':strings[i][:20],'gc': float(strings[i].count('G')+strings[i].count('C'))/len(strings[i]), 
	                 'cx':embedding[0,i], 'cy':embedding[1,i]} for i in xrange(N)]
	json.dump(data, open(outfile, 'w'))

if __name__ == "__main__":
	embedding, strings = embed(sys.argv[1])
	if len(sys.argv)==3:
		export_json(sys.argv[2],embedding, strings)
	plot_embedding(embedding)
