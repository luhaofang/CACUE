
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

lib.conv2d_backprop.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp]

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

def create_conv():
    outputchannel = 3
    channel = 3
    inputdim = 224
    kernelsize = 3
    stride = 3
    data = create_conv_data(channel, inputdim, inputdim)
    kernel = create_conv_kernel_data(outputchannel, channel, kernelsize)
    bias = create_conv_bias_data(outputchannel)
    data_ = codecs.open('../conv/feature_map.txt', 'w')
    for channels in data:
        for i in range(len(channels)):
            data_.write(pack('f', channels[i]))
    data_.close()

    kernel_ = codecs.open('../conv/kernel.txt', 'w')
    for output_channels in kernel:
        for channels in output_channels:
            for i in range(len(channels)):
                kernel_.write(pack('f', channels[i]))
    kernel_.close()

    bias_ = codecs.open('../conv/bias.txt', 'w')
    for i in range(len(bias)):
        bias_.write(pack('f', bias[i][0]))
    bias_.close()

    result = conv2d(data, kernel, bias, stride)
    result_ = codecs.open('../conv/conv_result.txt', 'w')
    for output_channels in result:
        for dim in output_channels:
            result_.write(pack('f', dim))
    result_.close()


def conv2d_backprop(data, filters, grad, stride):
    dataSize = int(numpy.sqrt(data.shape[1]))

    filterSize = int(numpy.sqrt(filters.shape[2]))

    outputSize = int(numpy.sqrt(grad.shape[1]))

    print dataSize,filterSize,((dataSize - filterSize) / stride + 1),outputSize
    assert(((dataSize - filterSize) / stride + 1) == outputSize)

    channelSize = int(filters.shape[1])

    nFilters = int(filters.shape[0])

    data = N.require(data, numpy.float32, ['ALIGNED'])
    filters = N.require(filters, numpy.float32, ['ALIGNED'])
    filters = N.require(filters, numpy.float32, ['ALIGNED'])

    f_grad = numpy.zeros((nFilters, channelSize , filterSize * filterSize), dtype=N.float32)
    b_grad = numpy.zeros((nFilters,1), dtype=N.float32)
    result = numpy.zeros((channelSize,  dataSize * dataSize), dtype=N.float32)

    print dataSize, filterSize , stride, outputSize

    lib.conv2d_backprop(grad, outputSize, nFilters, filters, filterSize, data, channelSize, stride, result, f_grad, b_grad, dataSize)

    return result,f_grad,b_grad

def create_conv_grad():
    outputchannel = 32
    channel = 3
    inputdim = 227
    kernelsize = 3
    stride = 1
    outputdim = (inputdim - kernelsize)/ stride + 1;
    data = create_conv_data(channel, inputdim, inputdim)
    grad = create_conv_data(outputchannel, outputdim, outputdim)
    kernel = create_conv_kernel_data(outputchannel, channel, kernelsize)

    data_ = codecs.open('../conv_grad/feature_map.txt', 'w')
    for channels in data:
        for i in range(len(channels)):
            data_.write(pack('f', channels[i]))
    data_.close()

    grad_ = codecs.open('../conv_grad/grad_map.txt', 'w')
    for channels in grad:
        for i in range(len(channels)):
            grad_.write(pack('f', channels[i]))
    grad_.close()

    kernel_ = codecs.open('../conv_grad/kernel.txt', 'w')
    for output_channels in kernel:
        for channels in output_channels:
            for i in range(len(channels)):
                kernel_.write(pack('f', channels[i]))
    kernel_.close()

    result,fgrad, bgrad = conv2d_backprop(data, kernel, grad, stride)
    result_ = codecs.open('../conv_grad/conv_grad_result.txt', 'w')
    for output_channels in result:
        for dim in output_channels:
            result_.write(pack('f', dim))
    result_.close()

    fgrad_ = codecs.open('../conv_grad/fgrad.txt', 'w')
    for output_channels in fgrad:
        for channels in output_channels:
            for i in range(len(channels)):
                fgrad_.write(pack('f', channels[i]))
    fgrad_.close()

    bgrad_ = codecs.open('../conv_grad/bgrad.txt', 'w')
    for i in range(len(bgrad)):
        bgrad_.write(pack('f', bgrad[i][0]))
    bgrad_.close()


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
    create_conv()
    create_conv_grad()
    #grad = N.array([[1,1,1,2,2,2,3,3,3],[1,1,1,2,2,2,3,3,3],[1,1,1,2,2,2,3,3,3]]).astype(N.float32)
    #data = N.array([[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5],[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5],[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4,5,5,5,5,5]]).astype(N.float32)
    #kernel = N.array([[[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]],[[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]],[[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1,1]]]).astype(N.float32)

    #result, fgrad, bgrad = conv2d_backprop(data, kernel, grad, 1)
    #print result
    #print fgrad
    #print bgrad