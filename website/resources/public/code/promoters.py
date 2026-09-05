import cPickle,gzip,bz2,numpy,os,re,sys,datetime,random,json
from modshogun import WeightedDegreeStringKernel, StringCharFeatures, DNA, MultidimensionalScaling, KernelDistance

def load(filename):
    try:
        f = bz2.BZ2File(filename, 'rb')
    except IOError, details:
        sys.stderr.write('File ' + filename + ' cannot be read\n')
        sys.stderr.write(details)
        return
    myobj = cPickle.load(f)
    f.close()
    return myobj

def embed(file='mml.pickle',N=500):
	strings = []
	print '%s reading %s' % (datetime.datetime.now(), file)
	file_contents = load(file)
	print '%s there are %d strings in %s' % (datetime.datetime.now(), len(file_contents['examples']), file)
	
	positives = numpy.where(numpy.array(file_contents['labels'])>0)[0]
	selected_idxs = random.sample(positives,N)
	for i in selected_idxs:
		strings.append(file_contents['examples'][i])
	
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
