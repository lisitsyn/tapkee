#!/usr/bin/python3.12

import lib.tapkee as tapkee
import numpy as np

def parse_reduction_method(method_name: str):
    return tapkee.parse_reduction_method(method_name)

if __name__=="__main__":
    parameters = tapkee.ParametersSet()
    method = parse_reduction_method('ra')
    assert(method.name == 'Random Projection')
    try:
        parse_reduction_method('unknown')
        assert(False)
    except:
        pass
    #tapkee.initialize().withParameters(parameters).embedUsing(np.array([[0,2],[3,1]]))
