
import numpy as N
import numpy
import os
import ctypes
import codecs
from struct import pack

import cv2 as cv

_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('convolution', _path)
lib.conv2d.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned')]

lib.pool2d.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                       N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned')]
lib.pool2d_backprop.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                                N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                                N.ctypeslib.c_intp,
                                N.ctypeslib.ndpointer(N.float32, flags='aligned')]


def create_conv_data(channel, height, width):
    array_ = N.random.normal(0, 0.1,(channel, height*width)).astype(N.float32)
    return array_


def create_conv_kernel_data(output_channel, channel, kernel_size):
    kernel_ = N.random.normal(0,0.1,(output_channel, channel, kernel_size * kernel_size)).astype(N.float32)
    return kernel_

def create_conv_bias_data(output_channel):
    bias = N.random.normal(0,0.1,(output_channel,1)).astype(N.float32)
    return bias


def conv2d(data, filters, biases, stride):
    dataSize = int(numpy.sqrt(data.shape[1]))

    filterSize = int(numpy.sqrt(filters.shape[2]))

    assert(int(data.shape[0]) == int(filters.shape[1]))

    channelSize = int(filters.shape[1])

    nFilters = filters.shape[0]

    data = N.require(data, numpy.float32, ['ALIGNED'])
    filters = N.require(filters, numpy.float32, ['ALIGNED'])

    resultSize = (dataSize - filterSize)/ stride + 1;
    print dataSize, filterSize , stride, resultSize

    result = numpy.zeros((nFilters, resultSize*resultSize), dtype=N.float32)

    lib.conv2d(data, dataSize, channelSize, filters, filterSize, nFilters, biases, stride, result)
    return result


def pool2d(data, poolSize):
    nChannels = data.shape[0]
    dataSize = data.shape[1]

    resultSize = dataSize - poolSize + 1;

    result = numpy.zeros((nChannels, resultSize, resultSize), dtype=N.float32)
    lib.pool2d(data, dataSize, nChannels, poolSize, result)
    return result


def pool2dBackProp(data, dNextLayer, poolSize):
    nChannels = data.shape[0]
    dataSize = data.shape[1]

    result = numpy.zeros_like(data, dtype=N.float32)
    lib.pool2d_backprop(data, dataSize, dNextLayer, nChannels, poolSize, result)
    return result


if __name__ == '__main__':
    data = create_conv_data(3, 224, 224)
    kernel = create_conv_kernel_data(3, 3, 3)
    bias = create_conv_bias_data(3)
    data_ = codecs.open('./feature_map.txt','w')
    for channels in data :
        for i in range(len(channels)):
            data_.write(pack('f',channels[i]))
    data_.close()

    kernel_ = codecs.open('./kernel.txt','w')
    for output_channels in kernel:
        for channels in output_channels:
            for i in range(len(channels)):
                kernel_.write(pack('f',channels[i]))
    kernel_.close()

    bias_ = codecs.open('./bias.txt','w')
    for i in range(len(bias)):
        bias_.write(pack('f',bias[i][0]))
    bias_.close()

    result = conv2d(data,kernel,bias,1)
    result_ = codecs.open('./conv_result.txt','w')
    for output_channels in result:
        for dim in output_channels:
            result_.write(pack('f',dim))




