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

#ifndef VGG_NET_H_
#define VGG_NET_H_



#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace mycnn;


layer_block* vgg_conv_relu(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* vgg_conv_relu_maxpooling(blob_base* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	layer *al = new layer(new data_args(output_channel, 2, 2, 0, 0));
	al->op(CACU_MAX_POOLING, l->get_oblob());
	clock_t end = clock();
	*lb << l << al;
	return lb;
}

layer_block* vgg_fc(blob_base* data, int output_channel, int kernel_size = 0, int stride = 0, int pad = 0, op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(new data_args(output_channel,0,0,0,0));
		l->op(CACU_INNERPRODUCT, data)->op(activation_op)->op(CACU_DROPOUT);
		clock_t end = clock();
		*lb << l;
		return lb;
	}

network* create_vgg_16_net(int batch_size,phase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1_1 = vgg_conv_relu(blob_, 64, 3, 1, 1);
	conv1_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv1_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv1_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv1_2 = vgg_conv_relu_maxpooling(conv1_1->get_oblob(), 64, 3, 1, 1);
	conv1_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv1_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv1_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv2_1 = vgg_conv_relu(conv1_2->get_oblob(), 128, 3, 1, 1);
	conv2_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv2_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv2_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv2_2 = vgg_conv_relu_maxpooling(conv2_1->get_oblob(), 128, 3, 1, 1);
	conv2_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv2_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv2_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv3_1 = vgg_conv_relu(conv2_2->get_oblob(), 256, 3, 1, 1);
	conv3_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv3_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv3_2 = vgg_conv_relu(conv3_1->get_oblob(), 256, 3, 1, 1);
	conv3_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv3_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv3_3 = vgg_conv_relu_maxpooling(conv3_2->get_oblob(), 256, 3, 1, 1);
	conv3_3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv3_3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3_3->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv4_1 = vgg_conv_relu(conv3_3->get_oblob(), 512, 3, 1, 1);
	conv4_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv4_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv4_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv4_2 = vgg_conv_relu(conv4_1->get_oblob(), 512, 3, 1, 1);
	conv4_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv4_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv4_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv4_3 = vgg_conv_relu_maxpooling(conv4_2->get_oblob(), 512, 3, 1, 1);
	conv4_3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv4_3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv4_3->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv5_1 = vgg_conv_relu(conv4_3->get_oblob(), 512, 3, 1, 1);
	conv5_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv5_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv5_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv5_2 = vgg_conv_relu(conv5_1->get_oblob(), 512, 3, 1, 1);
	conv5_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv5_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv5_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv5_3 = vgg_conv_relu_maxpooling(conv5_2->get_oblob(), 512, 3, 1, 1);
	conv5_3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv5_3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv5_3->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *fc6 = vgg_fc(conv5_3->get_oblob(),4096);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	fc6->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

	layer_block *fc7 = vgg_fc(fc6->get_oblob(),4096);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	fc7->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);


	if(phrase_ == train){
		layer_block *loss_ = loss_layer(fc7->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		loss_->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

		*net << conv1_1 << conv1_2 << conv2_1 << conv2_2 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << conv5_1 << conv5_2 << conv5_3 << fc6 << fc7 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(fc7->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1_1 << conv1_2 << conv2_1 << conv2_2 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << conv5_1 << conv5_2 << conv5_3 << fc6 << fc7 << predict_;
	}


	return net;
}




/*

network* create_vgg_16_dy_net(int batch_size,phrase_type phrase_)
{
	em_blob *blob_ = cacu_allocator::create_em_blob(batch_size, 3, 224, 224, phrase_);
	em_bin_blob *label_ = cacu_allocator::create_em_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1_1 = conv_layer_nopooling(blob_, 64, 3, 1, 1);
	conv1_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv1_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv1_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv1_2 = conv_layer_maxpooling_relu_first(conv1_1->get_oblob(), 64, 3, 1, 1);
	conv1_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv1_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv1_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv2_1 = conv_layer_nopooling(conv1_2->get_oblob(), 128, 3, 1, 1);
	conv2_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv2_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv2_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv2_2 = conv_layer_maxpooling_relu_first(conv2_1->get_oblob(), 128, 3, 1, 1);
	conv2_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv2_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv2_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv3_1 = conv_layer_nopooling(conv2_2->get_oblob(), 256, 3, 1, 1);
	conv3_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv3_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv3_2 = conv_layer_nopooling(conv3_1->get_oblob(), 256, 3, 1, 1);
	conv3_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv3_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv3_3 = conv_layer_maxpooling_relu_first(conv3_2->get_oblob(), 256, 3, 1, 1);
	conv3_3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv3_3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3_3->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv4_1 = conv_layer_nopooling(conv3_3->get_oblob(), 512, 3, 1, 1);
	conv4_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv4_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv4_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv4_2 = conv_layer_nopooling(conv4_1->get_oblob(), 512, 3, 1, 1);
	conv4_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv4_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv4_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv4_3 = conv_layer_maxpooling_relu_first(conv4_2->get_oblob(), 512, 3, 1, 1);
	conv4_3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv4_3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv4_3->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv5_1 = conv_layer_nopooling(conv4_3->get_oblob(), 512, 3, 1, 1);
	conv5_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv5_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv5_1->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv5_2 = conv_layer_nopooling(conv5_1->get_oblob(), 512, 3, 1, 1);
	conv5_2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv5_2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv5_2->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *conv5_3 = conv_layer_maxpooling_relu_first(conv5_2->get_oblob(), 512, 3, 1, 1);
	conv5_3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.01f);
	conv5_3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv5_3->layers(0)->get_op<convolution_op>(0)->get_weight(1)->set_decay(0);

	layer_block *fc6 = fc_layer(conv5_3->get_oblob(),4096);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian, 0.01f);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	fc6->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

	layer_block *fc7 = fc_layer(fc6->get_oblob(),4096);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian, 0.01f);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	fc7->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);


	if(phrase_ == train){
		layer_block *loss_ = loss_layer(fc7->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian, 0.01f);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		loss_->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);

		*net << conv1_1 << conv1_2 << conv2_1 << conv2_2 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << conv5_1 << conv5_2 << conv5_3 << fc6 << fc7 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(fc7->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1f);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1_1 << conv1_2 << conv2_1 << conv2_2 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << conv5_1 << conv5_2 << conv5_3 << fc6 << fc7 << predict_;
	}


	return net;
}

*/


#endif