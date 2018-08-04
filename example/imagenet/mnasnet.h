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
#ifndef MNASNET_H_
#define MNASNET_H_

#include "../cifar10/layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;

layer_block* dw_sep_conv_block(blob_base* data,int output_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *dw = new layer(new data_args(data->channel(), kernel_size, stride, pad, data->channel()));
	dw->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(data->channel());

	layer *sep = new layer(new data_args(output_channel, 1, 1, 0, data->channel()));
	sep->op(CACU_CONVOLUTION, dw->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	clock_t end = clock();
	*lb << dw << sep;
	return lb;
}


layer_block* ide_dw_sep_conv_block(blob_base* data,int dep_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *split = new layer();
	split->op(CACU_SPLIT, data ,new op_args(2));

	layer *ide = new layer(new data_args(dep_channel, 1, stride, 0, data->channel()));
	ide->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	layer *dw = new layer(new data_args(dep_channel, kernel_size, 1, pad, dep_channel));
	dw->op(CACU_CONVOLUTION, ide->get_oblob())->op(CACU_BATCH_NORMALIZE);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(dep_channel);

	layer *sep = new layer(new data_args(data->channel(), 1, 1, 0, dep_channel));
	sep->op(CACU_CONVOLUTION, dw->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	blobs *sum_data = new blobs();
	sum_data->push_back(sep->get_oblob());
	sum_data->push_back(split->get_oblobs()->at(1));

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE, sum_data);

	clock_t end = clock();
	*lb << split << ide << dw << sep << elemsum;
	return lb;
}

layer_block* ide_dw_sep_conv_nskip_block(blob_base* data, int output_channel, int dep_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *ide = new layer(new data_args(dep_channel, 1, stride, 0, data->channel()));
	ide->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	layer *dw = new layer(new data_args(dep_channel, kernel_size, 1, pad, dep_channel));
	dw->op(CACU_CONVOLUTION, ide->get_oblob())->op(CACU_BATCH_NORMALIZE);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(dep_channel);

	layer *sep = new layer(new data_args(output_channel, 1, 1, 0, dep_channel));
	sep->op(CACU_CONVOLUTION, dw->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	clock_t end = clock();
	*lb << ide << dw << sep;
	return lb;
}

layer_block* ide_dw_sep_conv_shortcut_block(blob_base* data,int output_channel, int dep_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *split = new layer();
	split->op(CACU_SPLIT, data ,new op_args(2));

	layer *ide = new layer(new data_args(dep_channel, 1, stride, 0, data->channel()));
	ide->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	ide->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	layer *dw = new layer(new data_args(dep_channel, kernel_size, 1, pad, dep_channel));
	dw->op(CACU_CONVOLUTION, ide->get_oblob())->op(CACU_BATCH_NORMALIZE);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	dw->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(dep_channel);

	layer *sep = new layer(new data_args(output_channel, 1, 1, 0, dep_channel));
	sep->op(CACU_CONVOLUTION, dw->get_oblob())->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	sep->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	layer *shortcut = new layer(new data_args(output_channel, 1, stride, 0, data->channel()));
	shortcut->op(CACU_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(msra);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	shortcut->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(1);

	blobs *sum_data = new blobs();
	sum_data->push_back(sep->get_oblob());
	sum_data->push_back(shortcut->get_oblob());

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE, sum_data);

	clock_t end = clock();
	*lb << split << ide << dw << sep << shortcut << elemsum;
	return lb;
}


network* create_mnasnet(int batch_size_,phase_type phase_)
{
	blob *blob_ = cacu_allocator::create_blob(batch_size_, 3, 224, 224, phase_);
	bin_blob *label_ = cacu_allocator::create_bin_blob(batch_size_, 1, 1, 1,phase_);

	blobs *input_datas_ = cacu_allocator::create_blobs();
	input_datas_->push_back(blob_);
	input_datas_->push_back(label_);

	network *net = new network(input_datas_);

	layer *conv_header = new layer(new data_args(32, 3, 2, 1, blob_->channel()));
	conv_header->op(CACU_CONVOLUTION, blob_);

	*net << conv_header;

	layer_block *sep_conv1 = dw_sep_conv_block(conv_header->get_oblob(), 16, 3, 1, 1);

	layer_block *mb_conv1_1 = ide_dw_sep_conv_block(sep_conv1->get_oblob(), 16, 3, 1, 1);
	layer_block *mb_conv1_2 = ide_dw_sep_conv_block(mb_conv1_1->get_oblob(), 16, 3, 1, 1);
	layer_block *mb_conv1_3 = ide_dw_sep_conv_shortcut_block(mb_conv1_2->get_oblob(), 24, 16, 3, 2, 1);

	*net << mb_conv1_1 << mb_conv1_2 << mb_conv1_3;

	layer_block *mb_conv2_1 = ide_dw_sep_conv_block(mb_conv1_3->get_oblob(), 24, 5, 1, 2);
	layer_block *mb_conv2_2 = ide_dw_sep_conv_block(mb_conv2_1->get_oblob(), 24, 5, 1, 2);
	layer_block *mb_conv2_3 = ide_dw_sep_conv_shortcut_block(mb_conv2_2->get_oblob(), 40, 24, 5, 2, 2);

	*net << mb_conv2_1 << mb_conv2_2 << mb_conv2_3;

	layer_block *mb_conv3_1 = ide_dw_sep_conv_block(mb_conv2_3->get_oblob(), 40, 5, 1, 2);
	layer_block *mb_conv3_2 = ide_dw_sep_conv_block(mb_conv3_1->get_oblob(), 40, 5, 1, 2);
	layer_block *mb_conv3_3 = ide_dw_sep_conv_shortcut_block(mb_conv3_2->get_oblob(), 80, 40, 5, 1, 2);

	*net << mb_conv3_1 << mb_conv3_2 << mb_conv3_3;

	layer_block *mb_conv4_1 = ide_dw_sep_conv_block(mb_conv3_3->get_oblob(), 80, 3, 1, 1);
	layer_block *mb_conv4_2 = ide_dw_sep_conv_shortcut_block(mb_conv4_1->get_oblob(), 96, 80, 3, 1, 1);

	*net << mb_conv4_1 << mb_conv4_2 ;

	layer_block *mb_conv5_1 = ide_dw_sep_conv_block(mb_conv4_2->get_oblob(), 96, 5, 1, 2);
	layer_block *mb_conv5_2 = ide_dw_sep_conv_block(mb_conv5_1->get_oblob(), 96, 5, 1, 2);
	layer_block *mb_conv5_3 = ide_dw_sep_conv_block(mb_conv5_2->get_oblob(), 96, 5, 1, 2);
	layer_block *mb_conv5_4 = ide_dw_sep_conv_shortcut_block(mb_conv5_3->get_oblob(), 192, 96, 5, 2, 2);

	*net << mb_conv5_1 << mb_conv5_2 << mb_conv5_3 << mb_conv5_4;

	layer_block *mb_conv6_1 = ide_dw_sep_conv_nskip_block(mb_conv5_4->get_oblob(), 320, 192, 3, 1, 1);

	*net << mb_conv6_1;

	layer *ave_pool = new layer(new data_args(320,7,1,0,0));
	ave_pool->op(CACU_AVERAGE_POOLING, mb_conv6_1->get_oblob());

	if(phase_ == train){
		layer_block *loss_ = loss_layer((blob*)ave_pool->get_oblob(), label_, 1000);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(msra);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << ave_pool << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer((blob*)ave_pool->get_oblob(), 1000);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(msra);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << ave_pool << predict_;
	}

	return net;
}


#endif
