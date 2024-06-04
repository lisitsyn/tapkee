import matplotlib.pyplot as plt
import numpy as np

def generate_data(type, N=1000, random_state=None):
	rng = np.random.RandomState(random_state)
	if type=='swissroll':
		tt = np.array((3*np.pi/2)*(1+2*rng.rand(N)))
		height = np.array((rng.rand(N)-0.5))
		X = np.array([tt*np.cos(tt), 10*height, tt*np.sin(tt)])
		return X, tt
	if type=='scurve':
		tt = np.array((3*np.pi*(rng.rand(N)-0.5)))
		height = np.array((rng.rand(N)-0.5))
		X = np.array([np.sin(tt), 10*height, np.sign(tt)*(np.cos(tt)-1)])
		return X, tt
	if type=='helix':
		tt = np.linspace(1,N,N).T / N
		tt = tt*2*np.pi
		X = np.r_[[(2+np.cos(8*tt))*np.cos(tt)],
			[(2+np.cos(8*tt))*np.sin(tt)],
			[np.sin(8*tt)]]
		return X, tt
	if type=='twinpeaks':
		X = rng.uniform(-1, 1, size=(N, 2))
		tt = np.sin(np.pi * X[:, 0]) * np.tanh(X[:, 1])
		tt += 0.1 * rng.normal(size=tt.shape)
		X = np.vstack([X.T, tt])
		return X, tt
	if type=='klein':
		u = rng.uniform(0, 2 * np.pi, N)
		v = rng.uniform(0, 2 * np.pi, N)
		x = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.cos(u)
		y = (2 + np.cos(u / 2) * np.sin(v) - np.sin(u / 2) * np.sin(2 * v)) * np.sin(u)
		z = np.sin(u / 2) * np.sin(v) + np.cos(u / 2) * np.sin(2 * v)

		noise = 0.01
		x += noise * rng.normal(size=x.shape)
		y += noise * rng.normal(size=y.shape)
		z += noise * rng.normal(size=z.shape)
		return np.vstack((x, y, z)), u

	raise Exception('Dataset is not supported')

def plot(data, embedded_data, colors='m', method=None):
	fig = plt.figure()
	fig.set_facecolor('white')

	ax_original = fig.add_subplot(121, projection='3d')
	scatter_original = ax_original.scatter(data[0], data[1], data[2], c=colors, cmap=plt.cm.Spectral, s=5, picker=True)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Original', fontsize=9)

	ax_embedding = fig.add_subplot(122)
	scatter_embedding = ax_embedding.scatter(embedded_data[0], embedded_data[1], c=colors, cmap=plt.cm.Spectral, s=5, picker=True)
	plt.axis('tight')
	plt.axis('off')
	plt.title('Embedding' + (' with ' + method) if method else '', fontsize=9, wrap=True)

	highlighted_points = []  # To store highlighted points

	# Function to highlight points on both plots
	def highlight(index):
		# Reset previous highlighted points
		for point in highlighted_points:
			point.remove()
		highlighted_points.clear()

		# Highlight the current point on both scatter plots
		point1 = ax_original.scatter([data[0][index]], [data[1][index]], [data[2][index]], color='white', s=25, edgecolor='black', zorder=3)
		point2 = ax_embedding.scatter([embedded_data[0][index]], [embedded_data[1][index]], color='white', s=25, edgecolor='black', zorder=3)
		highlighted_points.append(point1)
		highlighted_points.append(point2)
		fig.canvas.draw_idle()

	# Event handler for mouse motion
	def on_hover(event):
		if event.inaxes == ax_original:
			cont, ind = scatter_original.contains(event)
		elif event.inaxes == ax_embedding:
			cont, ind = scatter_embedding.contains(event)
		else:
			return

		if cont:
			index = ind['ind'][0]
			highlight(index)

	fig.canvas.mpl_connect('motion_notify_event', on_hover)

	plt.show()
