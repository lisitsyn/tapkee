#!/usr/bin/python3.12

import lib.tapkee as tapkee
import numpy as np

def parse_reduction_method(method_name: str):
    return tapkee.parse_reduction_method(method_name)

def test_exception_unknown_method():
    try:
        parse_reduction_method('unknown')
        assert(False)
    except:
        pass

if __name__=='__main__':
    test_exception_unknown_method()
    parameters = tapkee.ParametersSet()
    method = parse_reduction_method('ra')
    assert(method.name == 'Random Projection')
    parameters.add(tapkee.Parameter.create('dimension reduction method', method))
    # tapkee_input  = np.array([[0,2,4],[3,1,9],[5,7,6]])
    tapkee_input  = np.loadtxt('../tapkee_jmlr_benchmarks/data/aviris.dat')
    tapkee_output = tapkee.initialize().withParameters(parameters).embedUsing(tapkee_input)
    print('>> From:\n', tapkee_input);
    print('<< To:\n', tapkee_output.embedding)
