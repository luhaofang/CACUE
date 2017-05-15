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


layer_block* conv_maxpooling(blob* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data);
	layer *ml = new layer(output_channel, 3, 2);
	ml->op(CACU_MAX_POOLING, (blob*)l->get_oblob())->op(activation_op);
	clock_t end = clock();
	*lb << l << ml;
	return lb;
}

layer_block* conv_avepooling(blob* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	layer *al = new layer(output_channel, 3, 2);
	al->op(CACU_AVERAGE_POOLING, (blob*)l->get_oblob());
	clock_t end = clock();
	*lb << l << al;
	return lb;
}

layer_block* conv_nopooling(blob* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_nopooling_norelu(blob* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data);
	clock_t end = clock();
	*lb << l;
	return lb;
}

network* create_cifar_test_net(int batch_size,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_maxpooling(blob_, 32, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv1");

	layer_block *conv2 = conv_avepooling((blob*)conv1->get_oblob(), 64, 5, 1, 2);
	conv2->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	conv2->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv2");

	layer_block *conv3 = conv_avepooling((blob*)conv2->get_oblob(), 64, 5, 1, 2);
	conv3->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	conv3->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv3");

	/*
	layer_block *conv4 = conv_nopooling((blob*)conv3->get_oblob(), 32, 5, 1, 2);
	conv4->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	conv4->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv4");
	layer_block *conv5 = conv_avepooling((blob*)conv4->get_oblob(), 32, 5, 1, 2);
	conv5->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	conv5->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv5");
	*/
	layer_block *conv4 = conv_nopooling_norelu((blob*)conv3->get_oblob(), 10, 1, 1, 0);
	conv4->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	conv4->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv4");



//	layer *al = new layer();
//	al->op(CACU_P_INNERPRODUCT,conv4_1->get_oblob());
//	al->get_op<p_innerproduct_op>(0)->set_weight_init_type(msra);
//	al->get_op<p_innerproduct_op>(0)->set_bias_init_type(constant);

	if(phrase_ == train){
		layer_block *loss_ = loss_layer(conv4->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_DEBUG("loss");
		*net << conv1 << conv2 << conv2_1 << conv3 << conv3_1 << conv4_1 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(conv4->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_DEBUG("predict");
		*net << conv1 << conv2 << conv2_1 << conv3 << conv3_1 << conv4_1 << predict_;
	}

	return net;
}
