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


network* create_cifar_quick_net(int batch_size,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_layer_maxpooling(blob_, 32, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.0001);
	conv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv1");
	layer_block *conv2 = conv_layer_avgpooling_relu_first((blob*)conv1->get_oblob(), 32, 5, 1, 2);
	conv2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01);
	conv2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv2");
	layer_block *conv3 = conv_layer_avgpooling_relu_first((blob*)conv2->get_oblob(), 64, 5, 1, 2);
	conv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01);
	conv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv3");
	layer_block *fc6 = fc_layer_nodropout((blob*)conv3->get_oblob(), 64);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("fc6");

	/*
	if(phrase_ == train){
		layer_block *loss_ = loss_layer((blob*)fc6->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_DEBUG("loss");
		*net << conv1 << conv2 << conv3 << fc6 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)fc6->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_DEBUG("predict");
		*net << conv1 << conv2 << conv3 << fc6 << predict_;
	}
	*/
	layer_block *fc = fc_layer_nodropout((blob*)fc6->get_oblob(), 10);
	LOG_DEBUG("fc");
	*net << conv1 << conv2 << conv3 << fc6 << fc;
	return net;
}

network* create_cifar_quick_test_net(int batch_size,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_layer_nopooling(blob_, 16, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv1");

	layer_block *conv2 = conv_layer_nopooling((blob*)conv1->get_oblob(), 32, 5, 1, 2);
	conv2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv2");

	layer_block *conv3 = conv_layer_nopooling((blob*)conv2->get_oblob(), 32, 5, 1, 2);
	conv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv3");

	/*
	layer_block *conv4 = conv_layer_nopooling((blob*)conv3->get_oblob(), 16, 5, 1, 2);
	conv4->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv4->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv4");
	//*/
	layer_block *conv1X1 = conv_layer((blob*)conv3->get_oblob(), 10, 1, 1, 0);
	conv1X1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv1X1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	//conv1X1->layers(0)->get_op<convolution_op>(0)->set_group(8);
	LOG_DEBUG("conv1X1");

	layer_block *conv1_1 = conv_layer(blob_, 10, 1, 1, 0);
	conv1_1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(msra);
	conv1_1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);

	blobs *b = new blobs();
	b->push_back(conv1_1->get_oblob());
	b->push_back(conv1X1->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b)->op(CACU_RELU);

	/*
	layer_block *fc6 = pfc_layer_nodropout((blob*)conv1X1->get_oblob(), 10);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("fc6");
	//*/

	layer_block *fc6 = pfc_layer_nodropout((blob*)element_wise->get_oblob(), 10);
	fc6->layers(0)->get_op<p_inner_product_op>(0)->set_weight_init_type(msra);
	fc6->layers(0)->get_op<p_inner_product_op>(0)->set_bias_init_type(constant);
	//fc6->layers(0)->get_op<convolution_op>(0)->set_group(10);
	LOG_DEBUG("fc6");

	/*
	layer_block *fc7 = fc_layer_nodropout((blob*)fc6->get_oblob(), 10);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	fc7->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("fc7");
	//*/
	*net << conv1 << conv2 << conv3 << conv1_1 << conv1X1 << element_wise << fc6;// << fc7;

	if(phrase_ == train){
		layer_block *loss_ = loss_without_fc((blob*)fc6->get_oblob(), label_, 10);
		LOG_DEBUG("loss");
		*net << loss_;
	}
	else
	{
		layer_block *predict_ = predict_without_fc((blob*)fc6->get_oblob(), 10);
		LOG_DEBUG("predict");
		*net << predict_;
	}
	//*/
	//*net << conv1 << conv1_1 << conv2 << conv2_1 << conv3 << conv3_1 << fc6;
	return net;
}

network* create_cifar_quick_net_nofc(int batch_size,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size, 3, 32, 32, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_layer_maxpooling(blob_, 32, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.0001f);
	conv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv1");
	layer_block *conv2 = conv_layer_avgpooling_relu_first((blob*)conv1->get_oblob(), 32, 5, 1, 2);
	conv2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01f);
	conv2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv2");
	layer_block *conv3 = conv_layer_avgpooling_relu_first((blob*)conv2->get_oblob(), 64, 5, 1, 2);
	conv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01f);
	conv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv3");

	*net << conv1 << conv2 << conv3 ;

	return net;
}

network* create_cifar_quick_dy_net(int batch_size,phrase_type phrase_)
{
	em_blob *blob_ = cacu_allocator::create_em_blob(batch_size, 3, 32, 32, phrase_);

	em_bin_blob *label_ = cacu_allocator::create_em_bin_blob(batch_size, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_layer_maxpooling(blob_, 32, 5, 1, 2);
	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.0001f);
	conv1->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv1");
	layer_block *conv2 = conv_layer_avgpooling_relu_first(conv1->get_oblob(), 32, 5, 1, 2);
	conv2->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01f);
	conv2->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv2");
	layer_block *conv3 = conv_layer_avgpooling_relu_first(conv2->get_oblob(), 64, 5, 1, 2);
	conv3->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian,0.01f);
	conv3->layers(0)->get_op<convolution_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("conv3");
	layer_block *fc6 = fc_layer_nodropout(conv3->get_oblob(), 64);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1f);
	fc6->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	LOG_DEBUG("fc6");
	if(phrase_ == train){
		layer_block *loss_ = loss_layer(fc6->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1f);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_DEBUG("loss");
		*net << conv1 << conv2 << conv3 << fc6 << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(fc6->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(gaussian,0.1f);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		LOG_DEBUG("predict");
		*net << conv1 << conv2 << conv3 << fc6 << predict_;
	}

	return net;
}



