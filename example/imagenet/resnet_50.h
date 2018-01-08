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


layer_block* conv_shortcut_block(blob_base* data,int output_channel, int output_channel_s, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();

	layer *split = new layer();
	split->op(CACU_SPLIT,data,new args(2));

	layer *shortcut = new layer(output_channel, 1, s_stride, 0, data->height(), data->channel());
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	shortcut->get_op<convolution_op>(0)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l1 = new layer(output_channel, kernel_size, stride, pad, shortcut->get_oblob()->height(), shortcut->get_oblob()->channel());
	l1->op(CACU_CONVOLUTION, shortcut->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l2 = new layer(output_channel_s, 1, stride, 0, l1->get_oblob()->height(), l1->get_oblob()->channel());
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l3 = new layer(output_channel_s, 1, s_stride, 0, split->get_oblobs()->at(1)->height(), split->get_oblobs()->at(1)->channel());
	l3->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE);
	l3->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l3->get_op<convolution_op>(0)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(l2->get_oblob());
	b->push_back(l3->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b)->op(activation_op);

	*lb << split << shortcut << l1 << l2 << l3 << element_wise;
	return lb;
}

layer_block* identity_block(blob_base* data,int output_channel, int output_channel_s, int kernel_size, int stride, int s_stride,int pad,op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();

	layer *split = new layer();
	split->op(CACU_SPLIT,data,new args(2));

	layer *shortcut = new layer(output_channel, 1, s_stride, 0, split->get_oblobs()->at(0)->height(), split->get_oblobs()->at(0)->channel());
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	shortcut->get_op<convolution_op>(0)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l1 = new layer(output_channel, kernel_size, stride, pad, shortcut->get_oblob()->height(), shortcut->get_oblob()->channel());
	l1->op(CACU_CONVOLUTION, shortcut->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l1->get_op<convolution_op>(0)->set_is_use_bias(false);

	layer *l2 = new layer(output_channel_s, 1, stride, 0, l1->get_oblob()->height(), l1->get_oblob()->channel());
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0)->set_weight_init_type(msra);
	l2->get_op<convolution_op>(0)->set_is_use_bias(false);

	blobs *b = new blobs();
	b->push_back(split->get_oblobs()->at(1));
	b->push_back(l2->get_oblob());

	layer *element_wise = new layer();
	element_wise->op(CACU_SUM_ELEMWISE, b)->op(activation_op);

	*lb << split << shortcut << l1 << l2 << element_wise;
	return lb;
}

network* create_res50net(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);
	layer_block *conv2a = conv_shortcut_block(conv1->get_oblob(),64,256,3,1,1,1);
	layer_block *conv2b = identity_block(conv2a->get_oblob(),64,256,3,1,1,1);
	layer_block *conv2c = identity_block(conv2b->get_oblob(),64,256,3,1,1,1);

	*net << conv1 << conv2a << conv2b << conv2c;

	layer_block *conv3a = conv_shortcut_block(conv2c->get_oblob(),128,512,3,1,2,1);
	layer_block *conv3b = identity_block(conv3a->get_oblob(),128,512,3,1,1,1);
	layer_block *conv3c = identity_block(conv3b->get_oblob(),128,512,3,1,1,1);
	layer_block *conv3d = identity_block(conv3c->get_oblob(),128,512,3,1,1,1);

	*net << conv3a << conv3b << conv3c << conv3d;

	layer_block *conv4a = conv_shortcut_block(conv3d->get_oblob(),256,1024,3,1,2,1);
	layer_block *conv4b = identity_block(conv4a->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4c = identity_block(conv4b->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4d = identity_block(conv4c->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4e = identity_block(conv4d->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4f = identity_block(conv4e->get_oblob(),256,1024,3,1,1,1);

	*net << conv4a << conv4b << conv4c << conv4d << conv4e << conv4f;

	layer_block *conv5a = conv_shortcut_block(conv4f->get_oblob(),512,2048,3,1,2,1);
	layer_block *conv5b = identity_block(conv5a->get_oblob(),512,2048,3,1,1,1);
	layer_block *conv5c = identity_block(conv5b->get_oblob(),512,2048,3,1,1,1);

	*net << conv5a << conv5b << conv5c;

	layer *ave_pool = new layer(conv5c->get_oblob()->channel(),conv5c->get_oblob()->height(),1);
	ave_pool->op(CACU_AVERAGE_POOLING, conv5c->get_oblob());

	*net << ave_pool;

	if(phrase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);

		*net << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(constant);
		*net << predict_;
	}
	return net;
}


network* create_res50net_nofc(int batch_size_,phrase_type phrase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phrase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phrase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer_block *conv1 = conv_block_top(blob_, 64, 7, 2, 3, false);
	layer_block *conv2a = conv_shortcut_block(conv1->get_oblob(),64,256,3,1,1,1);
	layer_block *conv2b = identity_block(conv2a->get_oblob(),64,256,3,1,1,1);
	layer_block *conv2c = identity_block(conv2b->get_oblob(),64,256,3,1,1,1);

	*net << conv1 << conv2a << conv2b << conv2c;

	layer_block *conv3a = conv_shortcut_block(conv2c->get_oblob(),128,512,3,1,2,1);
	layer_block *conv3b = identity_block(conv3a->get_oblob(),128,512,3,1,1,1);
	layer_block *conv3c = identity_block(conv3b->get_oblob(),128,512,3,1,1,1);
	layer_block *conv3d = identity_block(conv3c->get_oblob(),128,512,3,1,1,1);

	*net << conv3a << conv3b << conv3c << conv3d;

	layer_block *conv4a = conv_shortcut_block(conv3d->get_oblob(),256,1024,3,1,2,1);
	layer_block *conv4b = identity_block(conv4a->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4c = identity_block(conv4b->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4d = identity_block(conv4c->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4e = identity_block(conv4d->get_oblob(),256,1024,3,1,1,1);
	layer_block *conv4f = identity_block(conv4e->get_oblob(),256,1024,3,1,1,1);

	*net << conv4a << conv4b << conv4c << conv4d << conv4e << conv4f;

	layer_block *conv5a = conv_shortcut_block(conv4f->get_oblob(),512,2048,3,1,2,1);
	layer_block *conv5b = identity_block(conv5a->get_oblob(),512,2048,3,1,1,1);
	layer_block *conv5c = identity_block(conv5b->get_oblob(),512,2048,3,1,1,1);

	*net << conv5a << conv5b << conv5c;

	layer *ave_pool = new layer(conv5c->get_oblob()->channel(),conv5c->get_oblob()->height(),1);
	ave_pool->op(CACU_AVERAGE_POOLING, conv5c->get_oblob());

	*net << ave_pool;

	return net;
}
