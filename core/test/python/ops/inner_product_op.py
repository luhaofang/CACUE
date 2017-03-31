
import numpy as N
import numpy
import os
import ctypes
import codecs
from struct import pack


_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('innerproduct', _path)
lib.innerproduct.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp,
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'),
                       N.ctypeslib.ndpointer(N.float32, flags='aligned')]

lib.innerproduct_backprop.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.ndpointer(N.float32, flags='aligned'),
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.ndpointer(N.float32, flags='aligned'),
                       N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.ndpointer(N.float32, flags='aligned')]

def create_ip(dataSize,outputChannel):
    w = N.random.normal(0.0,1.0,(outputChannel,dataSize)).astype(N.float32)
    bias = N.random.normal(0.0,1.0,(outputChannel,1)).astype(N.float32)
    data = N.random.normal(0.0,1.0,dataSize).astype(N.float32)

    result = N.zeros((outputChannel),dtype = N.float32)

    lib.innerproduct(data,dataSize,w,outputChannel,bias,result)

    #for i in range(outputChannel):
    #    result[i] = N.dot(w[i],data)

    data_ = codecs.open('../innerproduct/feature_map.txt', 'w')
    for i in range(len(data)):
        data_.write(pack('f', data[i]))
    data_.close()

    kernel_ = codecs.open('../innerproduct/w.txt', 'w')
    for output_channels in w:
        for channels in output_channels:
            kernel_.write(pack('f', channels))
    kernel_.close()

    bias_ = codecs.open('../innerproduct/bias.txt', 'w')
    for i in range(len(bias)):
        bias_.write(pack('f', bias[i][0]))
    bias_.close()

    result_ = codecs.open('../innerproduct/result.txt', 'w')
    for dim in result:
        result_.write(pack('f', dim))
    result_.close()


def create_ip_grad(dataSize,outputChannel):
    w = N.random.normal(0.0,1.0,(outputChannel,dataSize)).astype(N.float32)
    bias = N.random.normal(0.0,1.0,(outputChannel,1)).astype(N.float32)
    data = N.random.normal(0.0,1.0,dataSize).astype(N.float32)
    grad = N.random.normal(0.0,1.0,outputChannel).astype(N.float32)

    result = N.zeros(dataSize,dtype = N.float32)
    f_grad = N.zeros((outputChannel, dataSize),dtype = N.float32)
    b_grad = N.zeros(outputChannel,dtype = N.float32)

    lib.innerproduct_backprop(data,dataSize ,grad,w,outputChannel,f_grad ,bias, b_grad, result)

    data_ = codecs.open('../innerproduct/feature_map_bp.txt', 'w')
    for i in range(len(data)):
        data_.write(pack('f', data[i]))
    data_.close()

    grad_ = codecs.open('../innerproduct/grad_bp.txt', 'w')
    for i in range(len(grad)):
        grad_.write(pack('f', grad[i]))
    grad_.close()

    kernel_ = codecs.open('../innerproduct/w_bp.txt', 'w')
    for output_channels in w:
        for channels in output_channels:
            kernel_.write(pack('f', channels))
    kernel_.close()

    bias_ = codecs.open('../innerproduct/bias_bp.txt', 'w')
    for i in range(len(bias)):
        bias_.write(pack('f', bias[i][0]))
    bias_.close()

    f_grad_ = codecs.open('../innerproduct/fgrad_bp.txt', 'w')
    for output_channels in f_grad:
        for channels in output_channels:
            f_grad_.write(pack('f', channels))
    f_grad_.close()

    b_grad_ = codecs.open('../innerproduct/bgrad_bp.txt', 'w')
    for i in range(len(b_grad)):
        b_grad_.write(pack('f', b_grad[i]))
    b_grad_.close()

    result_ = codecs.open('../innerproduct/result_bp.txt', 'w')
    for dim in result:
        result_.write(pack('f', dim))
    result_.close()

if __name__ == '__main__':
    create_ip(128*7*7,64)
    create_ip_grad(512*7*7,64)