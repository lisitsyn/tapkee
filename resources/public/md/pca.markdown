Principal Component Analysis
----------------------------

The Principal Component Analysis is probably the oldest dimension reduction algorithm which comes
in various flavours today. The simplest 'version' of the PCA algorithm could look like that:

* Subtract mean feature vector from each feature vector of a set $ X = \{ x\_1, x\_2, \dots, x\_N \} $.

* Compute the covariance matrix $ C $ using all feature vectors.

* Find top $ t $ (desired dimension of embedded space) and form projection matrix $ P $ with eigenvectors
  as columns.

* Project the data with $ Y = P X $.
