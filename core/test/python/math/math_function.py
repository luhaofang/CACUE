import numpy as N
import numpy

import codecs
from struct import pack

def create_isaxb_data(length):
    inputdata = N.random.normal(0,1,length)
    index = N.zeros(length)
