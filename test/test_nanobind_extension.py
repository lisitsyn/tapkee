import lib.tapkee as tapkee
import numpy as np

def test_exception_unknown_method():
    try:
        tapkee.parse_reduction_method('unknown')
        assert(False)
    except:
        pass

def test_embed_neighbors_method_brute():
    data = np.random.randn(100, 3)
    result = tapkee.embed(data, method='isomap', neighbors_method='brute',
                          num_neighbors=15, target_dimension=2)
    assert result.shape == (3, 2)

def test_embed_neighbors_method_vptree():
    data = np.random.randn(100, 3)
    result = tapkee.embed(data, method='isomap', neighbors_method='vptree',
                          num_neighbors=15, target_dimension=2)
    assert result.shape == (3, 2)

def test_embed_eigen_method_dense():
    data = np.random.randn(100, 3)
    result = tapkee.embed(data, method='pca', eigen_method='dense',
                          target_dimension=2)
    assert result.shape == (3, 2)

def test_embed_eigen_method_randomized():
    data = np.random.randn(100, 3)
    result = tapkee.embed(data, method='pca', eigen_method='randomized',
                          target_dimension=2)
    assert result.shape == (3, 2)

def test_embed_unknown_neighbors_method():
    data = np.random.randn(100, 3)
    try:
        tapkee.embed(data, method='isomap', neighbors_method='nonexistent')
        assert False, "Should have raised an error"
    except RuntimeError:
        pass

def test_embed_unknown_eigen_method():
    data = np.random.randn(100, 3)
    try:
        tapkee.embed(data, method='pca', eigen_method='nonexistent')
        assert False, "Should have raised an error"
    except RuntimeError:
        pass

if __name__=='__main__':
    test_exception_unknown_method()
    test_embed_neighbors_method_brute()
    test_embed_neighbors_method_vptree()
    test_embed_eigen_method_dense()
    test_embed_eigen_method_randomized()
    test_embed_unknown_neighbors_method()
    test_embed_unknown_eigen_method()
    parameters = tapkee.ParametersSet()
    method = tapkee.parse_reduction_method('spe')
    assert(method.name == 'Stochastic Proximity Embedding (SPE)')
    parameters.add(tapkee.Parameter.create('dimension reduction method', method))
    target_dimension = 2
    parameters.add(tapkee.Parameter.create('target dimension', target_dimension))
    data = np.random.randn(124, 3)
    embedded_data = tapkee.withParameters(parameters).embedUsing(data).embedding
    assert(embedded_data.shape == tuple([data.shape[1], target_dimension]))
