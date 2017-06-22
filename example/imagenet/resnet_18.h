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

layer_block* conv_block_top(blob_base* data,int output_channel, int kernel_size, int stride, int pad, bool usebias,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l->get_op<convolution_op>(0)->set_is_use_bias(usebias);
	layer *ml = new layer(output_channel, 3, 2);
	ml->op(CACU_MAX_POOLING, (blob*)l->get_oblob());
	clock_t end = clock();
	*lb << l << ml;
	return lb;
}

layer_block* conv_block_shotcut(blob_base* data,int output_channel, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *split = new layer();
	split->op(CACU_SPLIT,data,new args(2));

	layer *shortcut = new layer(output_channel, 1, s_stride, 0, data->height(), data->channel());
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	shortcut->get_op<convolution_op>(0)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l1 = new layer(output_channel, kernel_size, s_stride, pad, data->height(), data->channel());
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l2 = new layer(output_channel, kernel_size, stride, pad, l1->get_oblob()->height(), l1->get_oblob()->channel());
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l2->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(shortcut->get_oblob());
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b);

	layer *split1 = new layer();
	split1->op(CACU_SPLIT,element_wise->get_oblob(),new args(2));

	layer *l3 = new layer(output_channel, kernel_size, stride, pad, split1->get_oblobs()->at(0)->height(), split1->get_oblobs()->at(0)->channel());
	l3->op(CACU_CONVOLUTION, split1->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l3->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l3->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l4 = new layer(output_channel, kernel_size, stride, pad, l3->get_oblob()->height(), l3->get_oblob()->channel());
	l4->op(CACU_CONVOLUTION, l3->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l4->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l4->get_op<convolution_op>(0)->set_is_use_bias(false);

	blobs *b1 = new blobs();
	b1->push_back(split1->get_oblobs()->at(1));
	b1->push_back(l4->get_oblob());

	layer *block_wise = new layer();
	block_wise->op(CACU_SUM_ELEMWISE, b1);

	clock_t end = clock();
	*lb << split << shortcut << l1 << l2 << element_wise << split1 << l3 << l4 << block_wise;
	return lb;
}

network* create_testnet(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);

	layer *conv2 = new layer(512,3,1,1);
	conv2->op(CACU_CONVOLUTION,conv1->get_oblob());
	conv2->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *ave_pool = new layer(64,conv2->get_oblob()->height(),1);
	ave_pool->op(CACU_AVERAGE_POOLING,conv2->get_oblob());

	if(phrase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << ave_pool << predict_;
	}


	return net;
}

network* create_res18net(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);
	layer_block *conv2 = conv_block_shotcut(conv1->get_oblob(),64,3,1,1,1);
	layer_block *conv3 = conv_block_shotcut(conv2->get_oblob(),128,3,1,2,1);
	layer_block *conv4 = conv_block_shotcut(conv3->get_oblob(),256,3,1,2,1);
	layer_block *conv5 = conv_block_shotcut(conv4->get_oblob(),512,3,1,2,1);

	layer *injector = new layer();
	injector->op(CACU_INJECTOR,conv5->get_oblob());

	layer *ave_pool = new layer(512,7,1);
	ave_pool->op(CACU_AVERAGE_POOLING,conv5->get_oblob());

	if(phrase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << predict_;
	}

	return net;
}

network* create_res18net_without_fc(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);
	layer_block *conv2 = conv_block_shotcut(conv1->get_oblob(),64,3,1,1,1);
	layer_block *conv3 = conv_block_shotcut(conv2->get_oblob(),128,3,1,2,1);
	layer_block *conv4 = conv_block_shotcut(conv3->get_oblob(),256,3,1,2,1);
	layer_block *conv5 = conv_block_shotcut(conv4->get_oblob(),512,3,1,2,1);

	layer *ave_pool = new layer(512,7,1);
	ave_pool->op(CACU_AVERAGE_POOLING,conv5->get_oblob());

	layer_block *fc8 = fc_layer_nodropout(ave_pool->get_oblob(),1000);
	fc8->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
	fc8->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
	fc8->layers(0)->get_op<inner_product_op>(0)->get_weight(1)->set_decay(0);


	*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << fc8;

	return net;
}

network* create_res18_dy_net(int batch_size_,phrase_type phrase_)
{
	em_blob *blob_ = cacu_allocator::create_em_blob(batch_size_, 3, 224, 224, phrase_);
	em_bin_blob *label_ = cacu_allocator::create_em_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);
	layer_block *conv2 = conv_block_shotcut(conv1->get_oblob(),64,3,1,1,1);
	layer_block *conv3 = conv_block_shotcut(conv2->get_oblob(),128,3,1,2,1);
	layer_block *conv4 = conv_block_shotcut(conv3->get_oblob(),256,3,1,2,1);
	layer_block *conv5 = conv_block_shotcut(conv4->get_oblob(),512,3,1,2,1);

	layer *ave_pool = new layer(512,7,1);
	ave_pool->op(CACU_AVERAGE_POOLING,conv5->get_oblob());

	if(phrase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << conv1 << conv2 << conv3 << conv4 << conv5 << ave_pool << predict_;
	}


	return net;
}
