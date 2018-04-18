/*
copyright (c) 2016, David lu
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

#ifndef CIFAR_TEST_NET_H_
#define CIFAR_TEST_NET_H_

#include "layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;


network* create_cifar_test_net(int batch_size, phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1, phase_);

	blob *mask = new blob(batch_size, 3, 32, 32, 0.05, phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	/*
	layer_block *tconv1 = conv_layer(blob_, 3, 3, 1, 1);
	tconv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.33);
	tconv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_INFO("tconv1");
	layer_block *tconv2 = conv_layer((blob*)tconv1->get_oblob(), 3, 3, 1, 1);
	tconv2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.013);
	tconv2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_INFO("tconv2");
	layer_block *tconv3 = conv_layer_noActi((blob*)tconv2->get_oblob(), 3, 3, 1, 1);
	tconv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.013);
	tconv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_INFO("tconv3");
	*/
	layer_block *tconv1 = conv_layer(blob_, 3, 3, 1);
	tconv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(xavier);
	tconv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	//tconv1->layers(0)->get_op<convolution_op>(0)->set_group(3);
	LOG_INFO("tconv1");
	layer_block *tconv2 = fc_layer((blob*)tconv1->get_oblob(), 34 * 34 * 3);
	tconv2->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	tconv2->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	//tconv2->layers(0)->get_op<convolution_op>(0)->set_group(3);
	LOG_INFO("tconv2");
	((blob*)tconv2->get_oblob())->resize(batch_size, 3, 34, 34);

	layer_block *tconv3 = conv_layer((blob*)tconv2->get_oblob(), 3, 3, 1, 0, CACU_HTANH);
	tconv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(xavier);
	tconv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	//tconv2->layers(0)->get_op<convolution_op>(0)->set_group(3);
	LOG_INFO("tconv3");
	//((blob*)tconv3->get_oblob())->resize(batch_size, 3, 32, 32);

	layer *mul = new layer();
	mul->op(CACU_MATH, blob_);
	mul->get_op<math_op>(0)->function(MUL, (blob*)tconv3->get_oblob());
	
	*net << tconv1 << tconv2 << tconv3 << mul;
	/*
	layer *split = new layer();
	split->op(CACU_SPLIT, sub->get_oblob(), new op_args(2));
	
	
	// for dimension constaint
	
	layer *dimension = new layer();
	dimension->op(CACU_MATH, split->get_oblobs()->at(0));
	dimension->get_op<math_op>(0)->function(MASK_LT);	

	*net << split << dimension;
	*/
	bool updateornot = false;

	layer_block *conv1 = conv_layer_maxpooling(mul->get_oblob(), 32, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.1);
	conv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv1->layers(0)->get_op<convolution_op>(0)->set_is_update_weight(updateornot);
	LOG_INFO("conv1");
	layer_block *conv2 = conv_layer_avgpooling_relu_first((blob*)conv1->get_oblob(), 32, 5, 1, 2);
	conv2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.1);
	conv2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv2->layers(0)->get_op<convolution_op>(0)->set_is_update_weight(updateornot);
	LOG_INFO("conv2");
	layer_block *conv3 = conv_layer_avgpooling_relu_first((blob*)conv2->get_oblob(), 64, 5, 1, 2);
	conv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.1);
	conv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	conv3->layers(0)->get_op<convolution_op>(0)->set_is_update_weight(updateornot);
	LOG_INFO("conv3");
	layer_block *fc6 = fc_layer_nodropout((blob*)conv3->get_oblob(), 64);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian, 0.1);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_is_update_weight(updateornot);
	LOG_INFO("fc6");

	///*
	if (phase_ == train) {
		layer_block *loss_ = loss_layer((blob*)fc6->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian, 0.1);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_is_update_weight(updateornot);
		loss_->layers(0)->get_op<softmax_with_loss_op>(1)->set_loss_weight(1);
		LOG_INFO("loss");
		*net << conv1 << conv2 << conv3 << fc6 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)fc6->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian, 0.1);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_INFO("predict");
		*net << conv1 << conv2 << conv3 << fc6 << predict_;
	}
	//*/
	//layer_block *fc = fc_layer_nodropout((blob*)fc6->get_oblob(), 10);
	//LOG_DEBUG("fc");
	//*net << conv1 << conv2 << conv3 << fc6 << fc;
	return net;
}



#endif
