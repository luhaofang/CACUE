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

#ifndef RESNET_18_H_
#define RESNET_18_H_

#include "../cifar10/layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;

layer_block* conv_block_top(blob_base* data,int output_channel, int kernel_size, int stride, int pad, bool usebias,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	layer *ml = new layer(new data_args(output_channel, 3, 2, 0, l->get_oblob()->channel()));
	ml->op(CACU_MAX_POOLING, l->get_oblob());
	clock_t end = clock();
	*lb << l << ml;
	return lb;
}

layer_block* conv_block20_top(blob_base* data,int output_channel, int kernel_size, int stride, int pad, bool usebias,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_shotcut(blob_base* data,int output_channel, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *split = new layer();
	split->op(CACU_SPLIT,data,new op_args(2));

	layer *shortcut = new layer(new data_args(output_channel, 1, s_stride, 0,data->channel()));
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l1 = new layer(new data_args(output_channel, kernel_size, s_stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, l1->get_oblob()->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(shortcut->get_oblob());
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b)->op(activation_op);

	layer *split1 = new layer();
	split1->op(CACU_SPLIT,element_wise->get_oblob(),new op_args(2));

	layer *l3 = new layer(new data_args(output_channel, kernel_size, stride, pad, split1->get_oblobs()->at(0)->channel()));
	l3->op(CACU_CONVOLUTION, split1->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l3->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l3->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l4 = new layer(new data_args(output_channel, kernel_size, stride, pad, l3->get_oblob()->channel()));
	l4->op(CACU_CONVOLUTION, l3->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l4->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l4->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	blobs *b1 = new blobs();
	b1->push_back(split1->get_oblobs()->at(1));
	b1->push_back(l4->get_oblob());

	layer *block_wise = new layer();
	block_wise->op(CACU_SUM_ELEMWISE, b1)->op(activation_op);

	clock_t end = clock();
	*lb << split << shortcut << l1 << l2 << element_wise << split1 << l3 << l4 << block_wise;
	return lb;
}

layer_block* conv_res_shortcut_block(blob_base* data,int output_channel, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *split = new layer();
	split->op(CACU_SPLIT,data,new op_args(2));

	layer *shortcut = new layer(new data_args(output_channel, 1, s_stride, 0,data->channel()));
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l1 = new layer(new data_args(output_channel, kernel_size, s_stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, l1->get_oblob()->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(shortcut->get_oblob());
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b)->op(activation_op);

	*lb << split << shortcut << l1 << l2 << element_wise;
	return lb;
}

layer_block* conv_res_shortcut_block_identity(blob_base* data,int output_channel, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *split = new layer();
	split->op(CACU_SPLIT,data,new op_args(2));

	layer *shortcut = new layer(new data_args(output_channel, 1, s_stride, 0,data->channel()));
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l1 = new layer(new data_args(output_channel, kernel_size, s_stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, l1->get_oblob()->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(shortcut->get_oblob());
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b);

	*lb << split << shortcut << l1 << l2 << element_wise;
	return lb;
}

layer_block* conv_res_block(blob_base* data,int output_channel, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *split = new layer();
	split->op(CACU_SPLIT,data,new op_args(2));

	layer *l1 = new layer(new data_args(output_channel, kernel_size, s_stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, l1->get_oblob()->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(split->get_oblobs()->at(0));
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b)->op(activation_op);

	*lb << split << l1 << l2 << element_wise;
	return lb;
}

layer_block* conv_res_block_identity(blob_base* data,int output_channel, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *split = new layer();
	split->op(CACU_SPLIT,data,new op_args(2));

	layer *l1 = new layer(new data_args(output_channel, kernel_size, s_stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, l1->get_oblob()->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(split->get_oblobs()->at(0));
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b);

	*lb << split << l1 << l2 << element_wise;
	return lb;
}


network* create_res18net(int batch_size_,phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);
	layer_block *conv2 = conv_block_shotcut(conv1->get_oblob(),64,3,1,1,1);
	layer_block *conv3 = conv_block_shotcut(conv2->get_oblob(),128,3,1,2,1);
	layer_block *conv4 = conv_block_shotcut(conv3->get_oblob(),256,3,1,2,1);
	layer_block *conv5 = conv_block_shotcut(conv4->get_oblob(),512,3,1,2,1);

	layer *ave_pool = new layer(new data_args(512,7,1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING,conv5->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << predict_;
	}

	return net;
}


network* create_res20net(int batch_size_,phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block20_top(blob_, 16, 3, 1, 1, false);

	layer_block *conv2_1 = conv_res_block(conv1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_2 = conv_res_block(conv2_1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_3 = conv_res_block(conv2_2->get_oblob(),16,3,1,1,1);

	layer_block *conv3_1 = conv_res_shortcut_block(conv2_3->get_oblob(),32,3,1,2,1);
	layer_block *conv3_2 = conv_res_block(conv3_1->get_oblob(),32,3,1,1,1);
	layer_block *conv3_3 = conv_res_block(conv3_2->get_oblob(),32,3,1,1,1);

	layer_block *conv4_1 = conv_res_shortcut_block(conv3_3->get_oblob(),64,3,1,2,1);
	layer_block *conv4_2 = conv_res_block(conv4_1->get_oblob(),64,3,1,1,1);
	layer_block *conv4_3 = conv_res_block(conv4_2->get_oblob(),64,3,1,1,1);

	layer *ave_pool = new layer(new data_args(64, 7, 1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING, conv4_3->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << ave_pool << predict_;
	}


	return net;
}



network* create_res20net_tall(int batch_size_,phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block20_top(blob_, 16, 3, 1, 1, false);

	layer_block *conv2_1 = conv_res_block(conv1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_2 = conv_res_block(conv2_1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_3 = conv_res_block(conv2_2->get_oblob(),16,3,1,1,1);
	layer_block *conv2_4 = conv_res_block(conv2_3->get_oblob(),16,3,1,1,1);
	layer_block *conv2_5 = conv_res_block(conv2_4->get_oblob(),16,3,1,1,1);
	layer_block *conv2_6 = conv_res_block(conv2_5->get_oblob(),16,3,1,1,1);

	layer_block *conv3_1 = conv_res_shortcut_block(conv2_6->get_oblob(),32,3,1,2,1);
	layer_block *conv3_2 = conv_res_block(conv3_1->get_oblob(),32,3,1,1,1);
	layer_block *conv3_3 = conv_res_block(conv3_2->get_oblob(),32,3,1,1,1);
	layer_block *conv3_4 = conv_res_block(conv3_3->get_oblob(),32,3,1,1,1);
	layer_block *conv3_5 = conv_res_block(conv3_4->get_oblob(),32,3,1,1,1);
	layer_block *conv3_6 = conv_res_block(conv3_5->get_oblob(),32,3,1,1,1);

	layer_block *conv4_1 = conv_res_shortcut_block(conv3_6->get_oblob(),64,3,1,2,1);
	layer_block *conv4_2 = conv_res_block(conv4_1->get_oblob(),64,3,1,1,1);
	layer_block *conv4_3 = conv_res_block(conv4_2->get_oblob(),64,3,1,1,1);
	layer_block *conv4_4 = conv_res_block(conv4_3->get_oblob(),64,3,1,1,1);
	layer_block *conv4_5 = conv_res_block(conv4_4->get_oblob(),64,3,1,1,1);
	layer_block *conv4_6 = conv_res_block(conv4_5->get_oblob(),64,3,1,1,1);

	layer *ave_pool = new layer(new data_args(64, 7, 1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING, conv4_6->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3
				<< conv2_4 << conv2_5 << conv2_6
				<< conv3_1 << conv3_2 << conv3_3
				<< conv3_4 << conv3_5 << conv3_6
				<< conv4_1 << conv4_2 << conv4_3
				<< conv4_4 << conv4_5 << conv4_6 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3
						<< conv2_4 << conv2_5 << conv2_6
						<< conv3_1 << conv3_2 << conv3_3
						<< conv3_4 << conv3_5 << conv3_6
						<< conv4_1 << conv4_2 << conv4_3
						<< conv4_4 << conv4_5 << conv4_6 << ave_pool << predict_;
	}


	return net;
}


network* create_resnet_tall(int batch_size_, int size, phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block20_top(blob_, 16, 3, 1, 1, false);
	*net << conv1;

	for(int i = 0; i < size; ++i){
		layer_block *conv2_1 = conv_res_block(net->output_blob(),16,3,1,1,1);
		*net << conv2_1;
	}

	layer_block *conv3_1 = conv_res_shortcut_block(net->output_blob(),32,3,1,2,1);
	*net << conv3_1;

	for(int i = 0; i < size - 1; ++i){
		layer_block *conv3_2 = conv_res_block(net->output_blob(),32,3,1,1,1);
		*net << conv3_2;
	}

	layer_block *conv4_1 = conv_res_shortcut_block(net->output_blob(),64,3,1,2,1);
	*net << conv4_1;

	for(int i = 0; i < size - 1; ++i){
		layer_block *conv4_2 = conv_res_block(net->output_blob(),64,3,1,1,1);
		*net << conv4_2;
	}

	layer *ave_pool = new layer(new data_args(64, 7, 1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING, net->output_blob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << ave_pool << predict_;
	}


	return net;
}


network* create_res20net_fat(int batch_size_, int fat_size, phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block20_top(blob_, 16 * fat_size, 3, 1, 1, false);

	layer_block *conv2_1 = conv_res_block(conv1->get_oblob(),16 * fat_size,3,1,1,1);
	layer_block *conv2_2 = conv_res_block(conv2_1->get_oblob(),16 * fat_size,3,1,1,1);
	layer_block *conv2_3 = conv_res_block(conv2_2->get_oblob(),16 * fat_size,3,1,1,1);

	layer_block *conv3_1 = conv_res_shortcut_block(conv2_3->get_oblob(),32 * fat_size,3,1,2,1);
	layer_block *conv3_2 = conv_res_block(conv3_1->get_oblob(),32 * fat_size,3,1,1,1);
	layer_block *conv3_3 = conv_res_block(conv3_2->get_oblob(),32 * fat_size,3,1,1,1);

	layer_block *conv4_1 = conv_res_shortcut_block(conv3_3->get_oblob(),64 * fat_size,3,1,2,1);
	layer_block *conv4_2 = conv_res_block(conv4_1->get_oblob(),64 * fat_size,3,1,1,1);
	layer_block *conv4_3 = conv_res_block(conv4_2->get_oblob(),64 * fat_size,3,1,1,1);

//	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);//0.236
//
//	conv2_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//
//	conv3_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_1->layers(3)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//
//	conv4_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_1->layers(3)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);

	layer *ave_pool = new layer(new data_args(64 * fat_size, 7, 1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING, conv4_3->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << ave_pool << predict_;
	}


	return net;
}


network* create_res20net_identity(int batch_size_,phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block20_top(blob_, 16, 3, 1, 1, false);

	layer_block *conv2_1 = conv_res_block_identity(conv1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_2 = conv_res_block_identity(conv2_1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_3 = conv_res_block_identity(conv2_2->get_oblob(),16,3,1,1,1);

	layer_block *conv3_1 = conv_res_shortcut_block_identity(conv2_3->get_oblob(),32,3,1,2,1);
	layer_block *conv3_2 = conv_res_block_identity(conv3_1->get_oblob(),32,3,1,1,1);
	layer_block *conv3_3 = conv_res_block_identity(conv3_2->get_oblob(),32,3,1,1,1);

	layer_block *conv4_1 = conv_res_shortcut_block_identity(conv3_3->get_oblob(),64,3,1,2,1);
	layer_block *conv4_2 = conv_res_block_identity(conv4_1->get_oblob(),64,3,1,1,1);
	layer_block *conv4_3 = conv_res_block_identity(conv4_2->get_oblob(),64,3,1,1,1);

//	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);//0.236
//
//	conv2_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//
//	conv3_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_1->layers(3)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//
//	conv4_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_1->layers(3)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);

	layer *ave_pool = new layer(new data_args(64, 7, 1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING, conv4_3->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_is_use_bias(false);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv3_1 << conv3_2 << conv3_3 << conv4_1 << conv4_2 << conv4_3 << ave_pool << predict_;
	}


	return net;
}


network* create_res32net(int batch_size_,phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 28, 28, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block20_top(blob_, 16, 3, 1, 1, true);

	layer_block *conv2_1 = conv_res_block(conv1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_2 = conv_res_block(conv2_1->get_oblob(),16,3,1,1,1);
	layer_block *conv2_3 = conv_res_block(conv2_2->get_oblob(),16,3,1,1,1);
	layer_block *conv2_4 = conv_res_block(conv2_3->get_oblob(),16,3,1,1,1);
	layer_block *conv2_5 = conv_res_block(conv2_4->get_oblob(),16,3,1,1,1);

	layer_block *conv3_1 = conv_res_shortcut_block(conv2_5->get_oblob(),32,3,1,2,1);
	layer_block *conv3_2 = conv_res_block(conv3_1->get_oblob(),32,3,1,1,1);
	layer_block *conv3_3 = conv_res_block(conv3_2->get_oblob(),32,3,1,1,1);
	layer_block *conv3_4 = conv_res_block(conv3_3->get_oblob(),32,3,1,1,1);
	layer_block *conv3_5 = conv_res_block(conv3_4->get_oblob(),32,3,1,1,1);

	layer_block *conv4_1 = conv_res_shortcut_block(conv3_3->get_oblob(),64,3,1,2,1);
	layer_block *conv4_2 = conv_res_block(conv4_1->get_oblob(),64,3,1,1,1);
	layer_block *conv4_3 = conv_res_block(conv4_2->get_oblob(),64,3,1,1,1);
	layer_block *conv4_4 = conv_res_block(conv4_3->get_oblob(),64,3,1,1,1);
	layer_block *conv4_5 = conv_res_block(conv4_4->get_oblob(),64,3,1,1,1);

//	conv1->layers(0)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_4->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_4->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_5->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//	conv2_5->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.118);
//
//	conv3_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_1->layers(3)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_4->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_4->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_5->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//	conv3_5->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.083);
//
//	conv4_1->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_1->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_1->layers(3)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_2->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_2->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_3->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_3->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_4->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_4->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_5->layers(1)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);
//	conv4_5->layers(2)->get_op<convolution_op>(0)->set_weight_init_type(gaussian, 0.059);


	layer *ave_pool = new layer(new data_args(64, 7, 1 ,0 ,0));
	ave_pool->op(CACU_AVERAGE_POOLING,conv4_5->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 10);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv2_4 << conv2_5
				<< conv3_1 << conv3_2 << conv3_3 << conv3_4 << conv3_5
				<< conv4_1 << conv4_2 << conv4_3 << conv4_4 << conv4_5 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 10);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << conv1 << conv2_1 << conv2_2 << conv2_3 << conv2_4 << conv2_5
				<< conv3_1 << conv3_2 << conv3_3 << conv3_4 << conv3_5
				<< conv4_1 << conv4_2 << conv4_3 << conv4_4 << conv4_5 << ave_pool << predict_;
	}

	return net;
}



#endif
