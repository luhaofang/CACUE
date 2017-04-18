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

layer_block* conv_block_maxpooling(blob_base* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	l->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01);
	l->get_op<convolution_op>(0)->set_bias_init_type(constant,0.1);
	l->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);
	layer *ml = new layer(output_channel, 3, 2);
	ml->op(CACU_MAX_POOLING, l->get_oblob());
	clock_t end = clock();
	*lb << l << ml;
	return lb;
}

layer_block* conv_block_nopooling(blob_base* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	clock_t end = clock();
	*lb << l;
	return lb;
}


network* create_alexnet(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 227, 227, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_maxpooling(blob_, 96, 11, 4, 2);
	layer_block *conv2 = conv_block_maxpooling(conv1->get_oblob(), 256, 5, 1, 2);
	layer_block *conv3 = conv_block_nopooling(conv2->get_oblob(), 384, 3, 1, 1);
	layer_block *conv4 = conv_block_nopooling(conv3->get_oblob(), 384, 3, 1, 1);
	layer_block *conv5 = conv_block_maxpooling(conv4->get_oblob(), 256, 3, 1, 1);

	layer_block *fc6 = fc_layer(conv5->get_oblob(),4096);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.005);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant,0.1);
	fc6->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

	layer_block *fc7 = fc_layer(fc6->get_oblob(),4096);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.005);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant,0.1);
	fc7->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

	if(phrase_ == train){
		layer_block *loss_ = loss_layer(fc7->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.01);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		loss_->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << fc6 << fc7 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(fc7->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1f);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << fc6 << fc7 << predict_;
	}


	return net;
}

network* create_dy_alexnet(int batch_size_,phrase_type phrase_)
{
	dy_blob *blob_ = cacu_allocator::create_dy_blob(batch_size_, 3, 227, 227, phrase_);
	dy_bin_blob *label_ = cacu_allocator::create_dy_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_maxpooling(blob_, 96, 11, 4, 2);
	layer_block *conv2 = conv_block_maxpooling(conv1->get_oblob(), 256, 5, 1, 2);
	layer_block *conv3 = conv_block_nopooling(conv2->get_oblob(), 384, 3, 1, 1);
	layer_block *conv4 = conv_block_nopooling(conv3->get_oblob(), 384, 3, 1, 1);
	layer_block *conv5 = conv_block_maxpooling(conv4->get_oblob(), 256, 3, 1, 1);

	layer_block *fc6 = fc_layer(conv5->get_oblob(),4096);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.005);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant,0.1);
	fc6->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

	layer_block *fc7 = fc_layer(fc6->get_oblob(),4096);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.005);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant,0.1);
	fc7->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

	if(phrase_ == train){
		layer_block *loss_ = loss_layer(fc7->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.01);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		loss_->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << fc6 << fc7 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(fc7->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1f);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << fc6 << fc7 << predict_;
	}


	return net;
}
