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



#include "../../mycnn.h"

#include <time.h>

using namespace mycnn;


layer_block* conv_cifar_shortcut(blob_base* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	clock_t end = clock();
	*lb << l;
	return lb;
}

network* create_cifar_testnet(int batch_size,phrase_type phrase_){

	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_cifar_shortcut(blob_, 192, 5, 1, 2);
	layer_block *conv1_1 = conv_cifar_shortcut(conv1->get_oblob(), 160, 1, 1);
	layer_block *conv1_2 = conv_cifar_shortcut(conv1_1->get_oblob(), 96, 1, 1);

	*net << conv1 << conv1_1 << conv1_2;

	layer *ml = new layer(96, 3, 2);
	ml->op(CACU_MAX_POOLING, conv1_2->get_oblob())->op(CACU_DROPOUT);

	*net << ml;

	layer_block *conv2 = conv_cifar_shortcut(ml->get_oblob(), 192, 5, 1, 2);
	layer_block *conv2_1 = conv_cifar_shortcut(conv2->get_oblob(), 192, 1, 1);
	layer_block *conv2_2 = conv_cifar_shortcut(conv2_1->get_oblob(), 192, 1, 1);

	*net << conv2 << conv2_1 << conv2_2;

	layer *al_1 = new layer(192, 3, 2);
	al_1->op(CACU_AVERAGE_POOLING, conv2_2->get_oblob())->op(CACU_DROPOUT);

	*net << al_1;

	layer_block *conv3 = conv_cifar_shortcut(al_1->get_oblob(), 192, 3, 1, 1);
	layer_block *conv3_1 = conv_cifar_shortcut(conv3->get_oblob(), 192, 1, 1);
	layer_block *conv3_2 = conv_cifar_shortcut(conv3_1->get_oblob(), 10, 1, 1);

	*net << conv3 << conv3_1 << conv3_2;

	layer *al_2 = new layer(10, 8, 1);
	al_2->op(CACU_AVERAGE_POOLING, conv3_2->get_oblob())->op(CACU_DROPOUT);

	*net << al_2;

	if(phrase_ == train){
		layer *loss_ = new layer(10);
		blobs *data_ = new blobs();
		data_->push_back(al_2->get_oblob());
		data_->push_back(label_);
		loss_->op(CACU_SOFTMAX_LOSS,data_);
		*net << loss_;
	}
	else
	{
		layer *predict_ = new layer(10);
		predict_->op(CACU_SOFTMAX, al_2->get_oblob());
		*net << predict_;
	}

	return net;

}


