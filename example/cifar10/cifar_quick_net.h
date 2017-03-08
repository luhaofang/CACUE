/*
Copyright (c) 2016, David lu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#include "../mycnn.h"

#include <time.h>

using namespace mycnn;


network* create_cifar_quick_net()
{
	blob *b = cacu_allocator::create_blob(1, 3, 32, 32, 1, train);
	weight *_b = new weight("train",128, 3, 32, 32,train);
	_b->set_init_type(gaussian,1);
#if __PARALLELTYPE__ == __GPU__
	CUDA_PRINT(_b->s_data(),1);
#else
	LOG_INFO("%f,%f",_b->s_data()[0],_b->s_data()[1]);
#endif
	network *net = new network();

	layer_block *conv1 = conv_layer_avgpooling(_b, 32, 5, 1, 2);
	LOG_DEBUG("conv1");
	layer_block *conv2 = conv_layer_avgpooling_relu_first((blob*)conv1->get_oblob(), 32, 5, 1, 2);
	LOG_DEBUG("conv2");
	layer_block *conv3 = conv_layer_avgpooling_relu_first((blob*)conv2->get_oblob(), 64, 5, 1, 2);
	LOG_DEBUG("conv3");
	layer_block *fc6 = fc_layer((blob*)conv3->get_oblob(),64);
	LOG_DEBUG("fc6");
	layer_block *fc7 = fc_layer((blob*)fc6->get_oblob(),10);
	LOG_DEBUG("fc7");
	*net << conv1 << conv2 << conv3 << fc6 << fc7;

	return net;
}
