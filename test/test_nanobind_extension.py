import lib.tapkee as tapkee
import numpy as np

def test_exception_unknown_method():
    try:
        tapkee.parse_reduction_method('unknown')
        assert(False)
    except:
        pass

if __name__=='__main__':
    test_exception_unknown_method()
    parameters = tapkee.ParametersSet()
    method = tapkee.parse_reduction_method('spe')
    assert(method.name == 'Stochastic Proximity Embedding (SPE)')
    parameters.add(tapkee.Parameter.create('dimension reduction method', method))
    target_dimension = 2
    parameters.add(tapkee.Parameter.create('target dimension', target_dimension))
    data = np.random.randn(124, 3)
    embedded_data = tapkee.initialize().withParameters(parameters).embedUsing(data).embedding
    assert(embedded_data.shape == tuple([data.shape[1], target_dimension]))
