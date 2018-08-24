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

#ifndef LENET_H_
#define LENET_H_

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;


network* create_lenet(int batch_size, phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 1, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer *conv1 = new layer(new data_args(20, 5, 1, 0, 1));
	conv1->op(CACU_CONVOLUTION, blob_);
	conv1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(xavier);
	conv1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(constant);

	layer *pool1 = new layer(new data_args(20, 2, 2, 0, 20));
	pool1->op(CACU_MAX_POOLING, conv1->get_oblob());

	layer *conv2 = new layer(new data_args(50, 5, 1, 0, 20));
	conv2->op(CACU_CONVOLUTION, pool1->get_oblob());
	conv2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(xavier);
	conv2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(constant);

	layer *pool2 = new layer(new data_args(50, 2, 2, 0, 50));
	pool2->op(CACU_MAX_POOLING, conv2->get_oblob());

	layer *fc1 = new layer(new data_args(500, 0, 0, 0, 50));
	fc1->op(CACU_INNERPRODUCT, pool2->get_oblob())->op(CACU_RELU);
	fc1->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(xavier);
	fc1->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

	*net << conv1 << pool1 << conv2 << pool2 << fc1;
	///*
	if(phase_ == train){
		layer *loss_ = new layer(new data_args(10, 0, 0, 0, 500));
		loss_->op(CACU_INNERPRODUCT, fc1->get_oblob())->op(CACU_SOFTMAX_LOSS, label_);
		loss_->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(xavier);
		loss_->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);
		loss_->get_op<softmax_with_loss_op>(1, CACU_SOFTMAX_LOSS)->set_loss_weight(0.01);
		LOG_INFO("loss");
		*net << loss_;
	}
	else
	{
		layer *predict = new layer(new data_args(10, 1, 1, 0, 500));
		predict->op(CACU_INNERPRODUCT, fc1->get_oblob())->op(CACU_SOFTMAX);
		predict->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(xavier);
		predict->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);
		LOG_INFO("predict");
		*net << predict;
	}

	//net->_INIT_NETWORK();
	
	return net;
}



#endif
