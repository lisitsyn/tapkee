#!/usr/bin/python3.12

import lib.tapkee as tapkee
import numpy as np
from utils import generate_data, plot

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
    method = parse_reduction_method('lle')
    parameters.add(tapkee.Parameter.create('dimension reduction method', method))
    data, colors = generate_data('swissroll')
    embedded_data = tapkee.initialize().withParameters(parameters).embedUsing(data).embedding
    plot(data, embedded_data.T, colors)
