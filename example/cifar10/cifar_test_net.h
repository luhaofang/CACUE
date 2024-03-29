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

#ifndef CIFAR_QUICK_NET_H_
#define CIFAR_QUICK_NET_H_

#include "layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;


network* deconv(blob_base* data, int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{

	network *lb = new network();

	layer *dc = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	dc->op<conv_transpose_op>(data);//->op<softmax_op>(); //->op<batch_normalize_op>()
	dc->get_op<conv_transpose_op>(0)->set_weight_init_type(xavier);
	dc->get_op<conv_transpose_op>(0)->set_is_use_bias(usebias);
	//dc->get_op<batch_normalize_op>(1)->set_is_use_global_stats(false);

	*lb << dc;
	return lb;
}


network* create_cifar_quick_net(int batch_size, phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);
	layer_block *conv1 = conv_layer_maxpooling(blob_, 32, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.0001);
	conv1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(constant);
	//conv1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);
	//conv1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_norm_weights(true);

	layer *split1 = new layer();
	split1->op(CACU_SPLIT, conv1->get_oblob(), new op_args(2));

	LOG_INFO("conv1");
	layer_block *conv2 = conv_layer_avgpooling_relu_first((blob*)split1->get_oblobs()->at(0), 32, 5, 1, 2);
	conv2->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.01);
	conv2->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(constant);
	//conv2->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);
	//conv2->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_norm_weights(true);
	LOG_INFO("conv2");

	layer *split2 = new layer();
	split2->op(CACU_SPLIT, conv2->get_oblob(), new op_args(2));

	layer_block *conv3 = conv_layer_avgpooling_relu_first((blob*)split2->get_oblobs()->at(0), 64, 5, 1, 2);
	conv3->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.01);
	conv3->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(constant);
	//conv3->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);
	//conv3->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_norm_weights(true);
	LOG_INFO("conv3");

	layer *split3 = new layer();
	split3->op(CACU_SPLIT, conv3->get_oblob(), new op_args(2));

	layer_block *fc6 = fc_layer_nodropout((blob*)split3->get_oblobs()->at(0), 64);
	fc6->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(gaussian, 0.1);
	fc6->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);
	//fc6->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_is_use_bias(false);
	//fc6->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_is_norm_weights(true);
	LOG_INFO("fc6");

	///*
	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)fc6->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(gaussian, 0.1);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);
		//loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_is_use_bias(false);
		//loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_is_norm_weights(true);
		LOG_INFO("loss");
		*net << conv1 << split1 << conv2 << split2 << conv3 << split3 << fc6 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)fc6->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(xavier);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);
		//predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_is_use_bias(false);
		//predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_is_norm_weights(true);
		LOG_INFO("predict");
		*net << conv1 << split1 << conv2 << split2 << conv3 << split3 << fc6 << predict_;
	}

	return net;
}




#endif
