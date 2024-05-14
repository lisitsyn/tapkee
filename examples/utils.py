import matplotlib.pyplot as plt
import numpy as np

def generate_data(type, N=1000):
	if type=='swissroll':
		tt = np.array((3*np.pi/2)*(1+2*np.random.rand(N)))
		height = np.array((np.random.rand(N)-0.5))
		X = np.array([tt*np.cos(tt), 10*height, tt*np.sin(tt)])
		return X,tt
	if type=='scurve':
		tt = np.array((3*np.pi*(np.random.rand(N)-0.5)))
		height = np.array((np.random.rand(N)-0.5))
		X = np.array([np.sin(tt), 10*height, np.sign(tt)*(np.cos(tt)-1)])
		return X,tt
	if type=='helix':
		tt = np.linspace(1,N,N).T / N
		tt = tt*2*np.pi
		X = np.r_[[(2+np.cos(8*tt))*np.cos(tt)],
		             [(2+np.cos(8*tt))*np.sin(tt)],
		             [np.sin(8*tt)]]
		return X,tt

	raise Exception('Dataset is not supported')

def plot(data, embedded_data, colors='m', method=None):
	fig = plt.figure()
	fig.set_facecolor('white')

	ax = fig.add_subplot(121, projection='3d')
	ax.scatter(data[0], data[1], data[2], c=colors, cmap=plt.cm.Spectral, s=5)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Original', fontsize=9)

	ax = fig.add_subplot(122)
	ax.scatter(embedded_data[0], embedded_data[1], c=colors, cmap=plt.cm.Spectral, s=5)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Embedding' + (' with ' + method) if method else '', fontsize=9)

	plt.show()

