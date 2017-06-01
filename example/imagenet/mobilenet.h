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

layer_block* conv_block_mobile(blob_base* data,int output_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l->get_op<convolution_op>(0)->set_is_use_bias(usebias);
	l->get_op<convolution_op>(0)->set_group(group);
	clock_t end = clock();
	*lb << l;
	return lb;
}

network* create_mobilenet(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_mobile(blob_, 32, 3, 2, 1);

	*net << conv1;

	layer_block *conv2_1dw = conv_block_mobile(conv1->get_oblob(), 32, 3, 1, 1, 32);
	layer_block *conv2_1sep = conv_block_mobile(conv2_1dw->get_oblob(), 64, 1, 1, 0);
	layer_block *conv2_2dw = conv_block_mobile(conv2_1sep->get_oblob(), 64, 3, 2, 1, 64);
	layer_block *conv2_2sep = conv_block_mobile(conv2_2dw->get_oblob(), 128, 1, 1, 0);

	*net << conv2_1dw << conv2_1sep << conv2_2dw << conv2_2sep;

	layer_block *conv3_1dw = conv_block_mobile(conv2_2sep->get_oblob(), 128, 3, 1, 1, 128);
	layer_block *conv3_1sep = conv_block_mobile(conv3_1dw->get_oblob(), 128, 1, 1, 0);
	layer_block *conv3_2dw = conv_block_mobile(conv3_1sep->get_oblob(), 128, 3, 2, 1, 128);
	layer_block *conv3_2sep = conv_block_mobile(conv3_2dw->get_oblob(), 256, 1, 1, 0);

	*net << conv3_1dw << conv3_1sep << conv3_2dw << conv3_2sep;

	layer_block *conv4_1dw = conv_block_mobile(conv3_2sep->get_oblob(), 256, 3, 1, 1, 256);
	layer_block *conv4_1sep = conv_block_mobile(conv4_1dw->get_oblob(), 256, 1, 1, 0);
	layer_block *conv4_2dw = conv_block_mobile(conv4_1sep->get_oblob(), 256, 3, 2, 1, 256);
	layer_block *conv4_2sep = conv_block_mobile(conv4_2dw->get_oblob(), 512, 1, 1, 0);

	*net << conv4_1dw << conv4_1sep << conv4_2dw << conv4_2sep;

	layer_block *conv5_1dw = conv_block_mobile(conv4_2sep->get_oblob(), 512, 3, 1, 1, 512);
	layer_block *conv5_1sep = conv_block_mobile(conv5_1dw->get_oblob(), 512, 1, 1, 0);

	*net << conv5_1dw << conv5_1sep;

	layer_block *conv5_2dw = conv_block_mobile(conv5_1sep->get_oblob(), 512, 3, 1, 1, 512);
	layer_block *conv5_2sep = conv_block_mobile(conv5_2dw->get_oblob(), 512, 1, 1, 0);

	*net << conv5_2dw << conv5_2sep;

	layer_block *conv5_3dw = conv_block_mobile(conv5_2sep->get_oblob(), 512, 3, 1, 1, 512);
	layer_block *conv5_3sep = conv_block_mobile(conv5_3dw->get_oblob(), 512, 1, 1, 0);

	*net << conv5_3dw << conv5_3sep;

	layer_block *conv5_4dw = conv_block_mobile(conv5_3sep->get_oblob(), 512, 3, 1, 1, 512);
	layer_block *conv5_4sep = conv_block_mobile(conv5_4dw->get_oblob(), 512, 1, 1, 0);

	*net << conv5_4dw << conv5_4sep;

	layer_block *conv5_5dw = conv_block_mobile(conv5_4sep->get_oblob(), 512, 3, 1, 1, 512);
	layer_block *conv5_5sep = conv_block_mobile(conv5_5dw->get_oblob(), 512, 1, 1, 0);

	*net << conv5_5dw << conv5_5sep;

	layer_block *conv5_6dw = conv_block_mobile(conv5_5sep->get_oblob(), 512, 3, 2, 1, 512);
	layer_block *conv5_6sep = conv_block_mobile(conv5_6dw->get_oblob(), 1024, 1, 1, 0);

	*net << conv5_6dw << conv5_6sep;

	layer_block *conv6dw = conv_block_mobile(conv5_6sep->get_oblob(), 1024, 3, 1, 1, 1024);
	layer_block *conv6sep = conv_block_mobile(conv6dw->get_oblob(), 1024, 1, 1, 0);

	*net << conv6dw << conv6sep;

	layer *ave_pool = new layer(1024,7,1);
	ave_pool->op(CACU_AVERAGE_POOLING,conv6sep->get_oblob());

	if(phrase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << ave_pool << predict_;
	}

	return net;
}

