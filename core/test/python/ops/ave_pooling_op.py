
import numpy as N
import numpy
import os
import ctypes
import codecs
from struct import pack


_path = os.path.dirname('__file__')
lib = N.ctypeslib.load_library('avepooling', _path)
lib.avepooling.argtypes = [N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp, N.ctypeslib.c_intp, N.ctypeslib.c_intp,
                            N.ctypeslib.c_intp, N.ctypeslib.ndpointer(N.float32, flags='aligned'), N.ctypeslib.c_intp]


def create_avepooling(dataSize, filterSize, stride, channel):

    data = N.random.normal(0.0, 1.0, (channel, dataSize * dataSize)).astype(N.float32)

    output_dim = (dataSize - filterSize) / stride + 1

    if not ((dataSize - filterSize) % stride) == 0:
        output_dim += 1

    print output_dim

    result = N.zeros((channel, output_dim * output_dim), dtype=N.float32)

    lib.avepooling(data, dataSize, channel, stride, filterSize, result, output_dim)

    data_ = codecs.open('../pooling/a_feature_map.txt', 'w')
    for d in data:
        for i in range(len(d)):
            data_.write(pack('f', d[i]))
    data_.close()

    result_ = codecs.open('../pooling/a_result.txt', 'w')
    for d in result:
        for dim in d:
            result_.write(pack('f', dim))
    result_.close()



if __name__ == '__main__':
    create_avepooling(224,3,1,3)