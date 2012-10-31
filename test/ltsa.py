from numpy import *
from scipy import *

def LTSA(data,k,target_dim):
	number_of_vectors = data.shape[1]
	number_of_features = data.shape[0]
	
	# get distances
	dist_matrix = zeros([number_of_vectors, number_of_vectors])	
	for i in range(number_of_vectors):
		for j in range(number_of_vectors):
			dist_matrix[i,j] = \
				linalg.norm(data[:,i]-data[:,j])
	
	# allocate W
	W = zeros([number_of_vectors,number_of_vectors])
	# compute local coordinates
	for i in range(number_of_vectors):
		distances_for_i = dist_matrix[:,i]
		# get neighbor indexes
		neigh_idxs_for_i = argsort(distances_for_i)[1:k+1]
		print neigh_idxs_for_i
		for ni in neigh_idxs_for_i:
			print data[:,i].dot(data[:,ni])
			# get local data
		local_X = X[:,neigh_idxs_for_i]
		#print 'Gram',(local_X.T).dot((local_X.T).T)
		print 'Gram',(local_X.T-local_X.T.mean(0)).dot((local_X.T-local_X.T.mean(0)).T)
		# compute SVD
		v = linalg.svd((local_X.T-local_X.T.mean(0)),full_matrices=False)[0]
		print v
		Gi = zeros([k,target_dim+1])
		Gi[:,1:] = v[:, :target_dim]
		Gi[:,0] = 1./sqrt(k)
		print 'Gi', Gi
		GiGiT = dot(Gi,Gi.T)
		nbrs_x, nbrs_y = meshgrid(neigh_idxs_for_i,neigh_idxs_for_i)
		W[nbrs_x,nbrs_y] -= GiGiT
		W[neigh_idxs_for_i,neigh_idxs_for_i] += 1
	eigenvalues, eigenvectors = linalg.eigh(W)
	return eigenvectors[:,1:target_dim+1].T
		
X = loadtxt('../input.dat').T
tt = loadtxt('../colormap.dat')

new_feats = LTSA(X,7,2)
print new_feats.shape

from pylab import *
from mpl_toolkits.mplot3d import Axes3D

fig = figure()
ax = fig.add_subplot(121,projection='3d')
cset = ax.scatter(X[0], X[1], X[2], s=10,c=tt,cmap=cm.Spectral)

subplot = fig.add_subplot(122)
subplot.scatter(new_feats[0,:],new_feats[1,:],c=tt, cmap=cm.Spectral)
show()
