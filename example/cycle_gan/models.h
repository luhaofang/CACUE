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

layer_block* conv_block_cycleGAN_l(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(CACU_LEAKY_RELU, new op_args(0.2));
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_scale_init_type(gaussian, 1, 0.02);
	l->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_is_use_global_stats(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_cycleGAN(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(CACU_RELU);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_scale_init_type(gaussian, 1, 0.02);
	l->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_is_use_global_stats(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* deconv_block_cycleGAN(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_DECONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(CACU_RELU);
	l->get_op<convolution_op>(0, CACU_DECONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l->get_op<convolution_op>(0, CACU_DECONVOLUTION)->set_is_use_bias(usebias);
	l->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_scale_init_type(gaussian, 1, 0.02);
	l->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_is_use_global_stats(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* residual_block_cycleGAN(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();

	layer *split = new layer();
	split->op(CACU_SPLIT, data,new op_args(2));

	clock_t start = clock();
	layer *l1 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l1->op(CACU_CONVOLUTION, split->get_oblobs()->at(0))->op(CACU_BATCH_NORMALIZE)->op(CACU_RELU);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l1->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_scale_init_type(gaussian, 1, 0.02);
	l1->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_is_use_global_stats(false);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l2->op(CACU_CONVOLUTION, l1->get_oblob())->op(CACU_BATCH_NORMALIZE);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian,0.02);
	l2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	l2->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_scale_init_type(gaussian, 1, 0.02);
	l2->get_op<batch_normalize_op>(1,CACU_BATCH_NORMALIZE)->set_is_use_global_stats(false);

	blobs *sum_data = new blobs();
	sum_data->push_back(l2->get_oblob());
	sum_data->push_back(split->get_oblobs()->at(1));

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);

	clock_t end = clock();
	*lb << split << l1 << l2 << elemsum;
	return lb;
}

layer_block* abse_loss_cycleGAN(blob_base* data1, blob_base *data2)
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

layer_block* mse_loss_cycleGAN(blob_base* data1, blob_base *data2)
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

layer_block* sigmoid_loss_cycleGAN(blob_base* data, blob_base *label)
{
	blobs *datas = new blobs();
	datas->push_back(data);
	datas->push_back(label);
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer();
	l->op(CACU_SIGMOID_LOSS, datas);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* create_generator_cycleGan(int batch_size, int dim, phase_type phase_)
{
	blob *blob_= new blob(batch_size,3,256,256,0,phase_);
	layer_block *generator = new layer_block();

	*generator << conv_block_cycleGAN(blob_, dim, 7, 1, 3);
	*generator << conv_block_cycleGAN(generator->get_oblob(), dim*2, 3, 2, 1);
	*generator << conv_block_cycleGAN(generator->get_oblob(), dim*4, 3, 2, 1);

	for(int i = 0 ; i < 9 ;++i)
		*generator << residual_block_cycleGAN(generator->get_oblob(), dim * 4, 3, 1, 1);

	*generator << deconv_block_cycleGAN(generator->get_oblob(), dim * 2, 4, 2, 1);

	*generator << deconv_block_cycleGAN(generator->get_oblob(), dim, 4, 2, 1);

	layer *conv = new layer(new data_args(3, 7, 1, 3, dim));
	conv->op(CACU_CONVOLUTION, generator->get_oblob())->op(CACU_TANH);
	conv->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.02);

	*generator << conv;

	return generator;
}


layer_block* create_discriminator_cycleGan(int batch_size, int dim, phase_type phase_)
{

	blob *blob_= new blob(batch_size,3,256,256,0,phase_);
	bin_blob *label_ = new bin_blob(batch_size,1,32,32,0,phase_);
	layer_block *discriminator = new layer_block();

	layer *conv1 = new layer(new data_args(dim, 4, 2, 1, blob_->channel()));
	conv1->op(CACU_CONVOLUTION,blob_)->op(CACU_LEAKY_RELU, new op_args(0.2));
	conv1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.02);

	*discriminator << conv1;

	*discriminator << conv_block_cycleGAN_l(conv1->get_oblob(), dim*2, 4, 2, 1);

	*discriminator << conv_block_cycleGAN_l(discriminator->get_oblob(), dim*4, 4, 2, 1);

	*discriminator << conv_block_cycleGAN_l(discriminator->get_oblob(), dim*8, 3, 1, 1);

	layer *conv = new layer(new data_args(1, 3, 1, 1, discriminator->get_oblob()->channel()));
	conv->op(CACU_CONVOLUTION,discriminator->get_oblob());//->op(CACU_SIGMOID);
	conv->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(gaussian, 0.02);

	*discriminator << conv;

	*discriminator << sigmoid_loss_cycleGAN(conv->get_oblob(), label_);

	return discriminator;
}



#endif /* EXAMPLE_CYCLE_GAN_MODELS_H_ */
