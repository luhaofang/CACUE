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

#ifndef EXAMPLE_CYCLE_GAN_MODELS_H_
#define EXAMPLE_CYCLE_GAN_MODELS_H_


#include "../cifar10/layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;

layer_block* conv_block(blob_base* data,int output_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_LEAKY_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(group);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* deconv_block(blob_base* data,int output_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_DECONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l->get_op<convolution_op>(0, CACU_DECONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l->get_op<convolution_op>(0, CACU_DECONVOLUTION)->set_is_use_bias(usebias);
	l->get_op<convolution_op>(0, CACU_DECONVOLUTION)->set_group(group);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* residual_block(blob_base* data,int output_channel, int kernel_size, int stride,int pad,int group = 1, op_name activation_op = CACU_RELU, bool usebias = false)
{
	layer_block *lb = new layer_block();

	layer *split = new layer();
	split->op(CACU_SPLIT, data,new op_args(2));

	clock_t start = clock();
	layer *l1 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(activation_op);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(group);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(group);

	blobs *sum_data = new blobs();
	sum_data->push_back(l2->get_oblob());
	sum_data->push_back(split->get_oblobs()->at(1));

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);

	clock_t end = clock();
	*lb << split << l1 << l2 << elemsum;
	return lb;
}

layer_block* abse_loss(blob_base* data1, blob_base *data2)
{

	blobs *datas = new blobs();
	datas->push_back(data1);
	datas->push_back(data2);
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer();
	l->op(CACU_ABSE_LOSS, datas);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* mse_loss(blob_base* data1, blob_base *data2)
{

	blobs *datas = new blobs();
	datas->push_back(data1);
	datas->push_back(data2);
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer();
	l->op(CACU_MSE_LOSS, datas);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* sigmoid_loss(blob_base* data, blob_base *label)
{

	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(1,0,0,0,data->channel()));
	l->op(CACU_INNERPRODUCT, data)->op(CACU_SIGMOID_LOSS, label);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* create_generator_cycleGan(int batch_size, int dim, phase_type phase_)
{
	blob *blob_= new blob(batch_size,3,256,256,0,phase_);
	layer_block *generator = new layer_block();

	layer_block *block1 = conv_block(blob_, dim, 7, 1, 2, CACU_RELU);
	layer_block *block2 = conv_block(block1->get_oblob(), dim*2, 3, 2, 1, CACU_RELU);
	layer_block *block3 = conv_block(block2->get_oblob(), dim*4, 3, 2, 1, CACU_RELU);

	*generator << block1 << block2 << block3;

	for(int i = 0 ; i < 9 ;++i)
	{
		layer_block *resblock = residual_block(generator->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
		*generator << resblock;
	}
	/*
	layer_block *resblock1 = residual_block(block3->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock2 = residual_block(resblock1->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock3 = residual_block(resblock2->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock4 = residual_block(resblock3->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock5 = residual_block(resblock4->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock6 = residual_block(resblock5->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock7 = residual_block(resblock6->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock8 = residual_block(resblock7->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	layer_block *resblock9 = residual_block(resblock8->get_oblob(), dim * 4, 3, 1, 1, CACU_RELU);
	*generator << resblock1 << resblock2 << resblock3;
	*generator << resblock4 << resblock5 << resblock6;
	*generator << resblock7 << resblock8 << resblock9;
	*/
	layer_block *deconvblock1 = deconv_block(generator->get_oblob(), dim * 2, 3, 2, 1, CACU_RELU);
	layer_block *deconvblock2 = deconv_block(generator->get_oblob(), dim, 3, 2, 1, CACU_RELU);

	*generator << deconvblock1 << deconvblock2;

	layer *conv = new layer(new data_args(3, 7 ,1, 1, dim));
	conv->op(CACU_CONVOLUTION,blob_)->op(CACU_TANH);
	conv->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.02);
	conv->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	*generator << conv;

	return generator;
}


layer_block* create_discriminator_cycleGan(int batch_size, int dim, phase_type phase_)
{

	blob *blob_= new blob(batch_size,3,256,256,0,phase_);
	layer_block *discriminator = new layer_block();

	layer *conv1 = new layer(new data_args(dim, 4, 2, 1, blob_->channel()));
	conv1->op(CACU_CONVOLUTION,blob_)->op(CACU_LEAKY_RELU);
	conv1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.02);
	conv1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	layer_block *block1 = conv_block(conv1->get_oblob(), dim*2, 4, 2, 1, CACU_LEAKY_RELU);
	layer_block *block2 = conv_block(block1->get_oblob(), dim*4, 4, 2, 1, CACU_LEAKY_RELU);
	layer_block *block3 = conv_block(block2->get_oblob(), dim*8, 4, 1, 1, CACU_LEAKY_RELU);

	*discriminator << conv1 << block1 << block2 << block3;

	layer *sigmoid = new layer(new data_args(1,0,0,0,0));
	sigmoid->op(CACU_INNERPRODUCT, block3->get_oblob())->op(CACU_SIGMOID);
	sigmoid->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->set_weight_init_type(gaussian,0.01);
	sigmoid->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->set_bias_init_type(constant);
	sigmoid->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->get_weight(1)->set_decay(0);

	return discriminator;
}



#endif /* EXAMPLE_CYCLE_GAN_MODELS_H_ */
