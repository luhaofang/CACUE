import numpy as np
from struct import pack
import caffe
import codecs

root = '/home/seal/4T/shortv/models/'
#root = '/home/seal/caffe/models/bvlc_alexnet/'

def load():
    # Load the net
    caffe.set_mode_cpu()
    # You may need to train this caffemodel first
    # There should be script to help you do the training
    net = caffe.Net(root + 'deploy.prototxt', root + 'VGG16_v2.caffemodel', caffe.TEST)
    file_ = codecs.open('/home/seal/4T/cacue/imagenet/vgg16net.caffemodel','w')
    for key in net.params.keys():
    	w = net.params[key][0].data
	b = net.params[key][1].data
	if len(np.shape(w)) == 4:
	    output_weight(file_,w)
	else:
	    output_weight_fc(file_,w)
	output_weight_bias(file_,b)
    file_.close()
    
def output_weight(file_,param):
    shape = np.shape(param)
    print shape
    length = shape[0]*shape[1]*shape[2]*shape[3]
    file_.write(pack('i', length));
    count = 0
    for n in range(shape[0]):
	for c in range(shape[1]):
	    for h in range(shape[2]):
		for w in range(shape[3]):
		    if count < 10:
			print str(param[n][c][h][w])+','
		    count += 1
		    file_.write(pack('f', param[n][c][h][w]))

def output_weight_fc(file_,param):
    shape = np.shape(param)
    print shape
    length = shape[0]*shape[1]
    file_.write(pack('i', length));
    for n in range(shape[0]):
	for c in range(shape[1]):
	    file_.write(pack('f', param[n][c]))

def output_weight_bias(file_,param):
    shape = np.shape(param)
    print shape
    length = shape[0]
    file_.write(pack('i', length));
    count = 0
    for n in range(shape[0]):
	if count < 10:
	    print str(param[n])+','
	count += 1
	file_.write(pack('f', param[n]))


if __name__ == "__main__":
    # You will need to change this path
    load()
    print 'Caffemodel loaded and written to .mat files successfully!'
