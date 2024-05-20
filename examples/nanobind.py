import lib.tapkee as tapkee
from utils import generate_data, plot

if __name__=='__main__':
    parameters = tapkee.ParametersSet()
    method = tapkee.parse_reduction_method('lle')
    parameters.add(tapkee.Parameter.create('dimension reduction method', method))
    data, colors = generate_data('swissroll')
    embedded_data = tapkee.initialize().withParameters(parameters).embedUsing(data).embedding
    plot(data, embedded_data.T, colors)
