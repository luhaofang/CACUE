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

#ifndef EXAMPLE_STYLE_FACE_MODELS_H_
#define EXAMPLE_STYLE_FACE_MODELS_H_



#include "../cifar10/layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>
#include <math.h>

using namespace cacu;

#define C_BN_NUM 8
#define std_size 256
#define random_size 512

layer_block* conv_block(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	layer *l1 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l1->op<equalized_convolution_op>(data);
	l1->get_op<equalized_convolution_op>(0)->std_init_w();
	l1->get_op<equalized_convolution_op>(0)->set_is_use_bias(usebias);

	*lb << l1;
	return lb;
}

layer_block* conv_block_with_pooling(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	layer *l1 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l1->op<equalized_convolution_op>(data)->op<relu_op>();
	l1->get_op<equalized_convolution_op>(0)->std_init_w();
	l1->get_op<equalized_convolution_op>(0)->set_is_use_bias(usebias);

	layer *pooling = new layer(new data_args(l1->get_oblob()->channel(), 3, 2, 0, l1->get_oblob()->channel()));
	pooling->op<max_pooling_op>(l1->get_oblob());

	*lb << l1 << pooling;

	return lb;
}


layer_block* gblock_in(blob_base* data, bin_blob *label, int output_channel, bool is_upsampling=true)
{
	layer_block *lb = new layer_block();

	layer *split = new layer();
	split->op<split_op>(data, new op_args(2));

	*lb << split;

	layer *l1;
	l1 = new layer(new data_args(output_channel, 3, 2, 0, data->channel()));
	l1->op<equalized_conv_transpose_op>(split->get_oblobs()->at(0))->op<leaky_relu_op>(new op_args(0.2))->op<conditional_in_op>(label, new op_args(C_BN_NUM));
	l1->get_op<equalized_conv_transpose_op>(0)->set_use_wscale(true);
	l1->get_op<equalized_conv_transpose_op>(0)->std_init_w();
	l1->get_op<equalized_conv_transpose_op>(0)->set_is_use_bias(false);
	*lb << l1;

	layer *l2 = new layer(new data_args(output_channel, 3, 1, 1, output_channel));
	l2->op<equalized_convolution_op>(l1->get_oblob())->op<leaky_relu_op>(new op_args(0.2))->op<conditional_in_op>(label, new op_args(C_BN_NUM));
	l2->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l2->get_op<equalized_convolution_op>(0)->std_init_w();
	l2->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	*lb << l2;

	layer *split1 = new layer();
	split1->op<split_op>(l2->get_oblob(), new op_args(2));

	*lb << split1;

	layer *thresholds1 = new layer();
	thresholds1->op<scalex_op>(split1->get_oblobs()->at(0), new op_args(0));
	thresholds1->get_op<scalex_op>(0)->set_op_name("resGblock_out");
	*lb << thresholds1;

	layer *thresholds2 = new layer();
	thresholds2->op<scalex_op>(split1->get_oblobs()->at(1), new op_args(1));
	*lb << thresholds2;

	layer *l3 = new layer(new data_args(3, 1, 1, 0, output_channel));
	l3->op<equalized_convolution_op>(thresholds2->get_oblob())->op<scalex_op>(new op_args(0));
	l3->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l3->get_op<equalized_convolution_op>(0)->std_init_w();
	l3->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	l3->get_op<equalized_convolution_op>(0)->set_op_name("upsampler_out");

	*lb << l3;

	layer *upsample = new layer(new data_args(data->channel(), 2, 2, 0, data->channel()));
	upsample->op<un_pooling_op>(split->get_oblobs()->at(1));
	*lb << upsample;

	layer *scale = new layer(new data_args(3, 1, 1, 0, data->channel()));
	scale->op<equalized_convolution_op>(upsample->get_oblob())->op<scalex_op>(new op_args(1));
	scale->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	scale->get_op<equalized_convolution_op>(0)->std_init_w();
	scale->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	scale->get_op<equalized_convolution_op>(0)->set_op_name("upsampler_in");
//	scale->get_op<equalized_convolution_op>(0)->set_is_variable_weight(false);
	*lb << scale;

	blobs *indata = new blobs();
	indata->push_back(l3->get_oblob());
	indata->push_back(scale->get_oblob());

	layer *elemsum = new layer();
	elemsum->op<sum_elemwise_op>(indata);
	elemsum->get_op<sum_elemwise_op>(0)->set_op_name("Gout");

	*lb << elemsum;

	return lb;
}


layer_block* gblock_in_head(blobs* data, int output_channel, bool is_upsampling=false)
{
	layer_block *lb = new layer_block();

	layer *l1 = new layer(new data_args(output_channel, 1, 1, 0, output_channel));
	l1->op<conditional_in_op>(data, new op_args(C_BN_NUM));


	*lb << l1;

	layer *l2 = new layer(new data_args(output_channel, 3, 1, 1, output_channel));
	l2->op<equalized_convolution_op>(l1->get_oblob())->op<leaky_relu_op>(new op_args(0.2))->op<conditional_in_op>(data->at(1), new op_args(C_BN_NUM));
	l2->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l2->get_op<equalized_convolution_op>(0)->std_init_w();
	l2->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);

	*lb << l2;

	return lb;
}


layer_block* d_first_block(blob_base* data, int output_channel)
{
	layer_block *lb = new layer_block();

	layer *l1 = new layer(new data_args(output_channel, 1, 1, 0, data->channel()));
	l1->op<equalized_convolution_op>(data)->op<leaky_relu_op>(new op_args(0.2));
	l1->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l1->get_op<equalized_convolution_op>(0)->std_init_w();
	l1->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	*lb << l1;
	return lb;
}



layer_block* dblock(blob_base* data, int output_channel, bool is_downsampling = false)
{
	blob *blob_= new blob(data->num(), 3, data->width(), data->height(), 0, data->phase());

	layer_block *lb = new layer_block();

	layer *split = new layer();
	split->op<split_op>(blob_, new op_args(2));
	split->get_op<split_op>(0)->set_op_name("Din");

	*lb << split;

	layer *l0 = new layer(new data_args(data->channel(), 1, 1, 0, 3));
	l0->op<equalized_convolution_op>(split->get_oblobs()->at(0))->op<scalex_op>(new op_args(1));
	l0->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l0->get_op<equalized_convolution_op>(0)->std_init_w();
	l0->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	l0->get_op<equalized_convolution_op>(0)->set_op_name("downsampler_out");
	//l0->get_op<scalex_op>(1)->set_op_name("dthreshold");
	*lb << l0;

	layer *sum = new layer();
	sum->op<scalex_op>(data, new op_args(0))->op<sum_elemwise_op>(l0->get_oblob());
	sum->get_op<sum_elemwise_op>(1)->set_op_name("dthreshold");
	*lb << sum;

	layer *l1 = new layer(new data_args(output_channel, 3, 1, 1, data->channel()));
	l1->op<equalized_convolution_op>(sum->get_oblob())->op<leaky_relu_op>(new op_args(0.2));
	l1->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l1->get_op<equalized_convolution_op>(0)->std_init_w();
	l1->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	*lb << l1;

	layer *l2;
	if(is_downsampling){
		l2 = new layer(new data_args(output_channel, 2, 2, 0, output_channel));
		l2->op<average_pooling_op>(l1->get_oblob())->op<leaky_relu_op>(new op_args(0.2))->op<scalex_op>(new op_args(0));
	}
	else{
		l2 = new layer(new data_args(output_channel, 3, 2, 1, output_channel));
		l2->op<equalized_convolution_op>(l1->get_oblob())->op<leaky_relu_op>(new op_args(0.2))->op<scalex_op>(new op_args(0));
		l2->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
		l2->get_op<equalized_convolution_op>(0)->std_init_w();
		l2->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	}
	*lb << l2;

	layer *downsample = new layer(new data_args(3, 2, 2, 0, 3));
	downsample->op<average_pooling_op>(split->get_oblobs()->at(1));

	*lb << downsample;

	layer *scale = new layer(new data_args(output_channel, 1, 1, 0, 3));
	scale->op<equalized_convolution_op>(downsample->get_oblob())->op<scalex_op>(new op_args(1));
	scale->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	scale->get_op<equalized_convolution_op>(0)->std_init_w();
	scale->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	scale->get_op<equalized_convolution_op>(0)->set_op_name("downsampler_in");
//	scale->get_op<equalized_convolution_op>(0)->set_is_variable_weight(false);

	*lb << scale;

	blobs *indata = new blobs();
	indata->push_back(l2->get_oblob());
	indata->push_back(scale->get_oblob());

	layer *elemsum = new layer();
	elemsum->op<sum_elemwise_op>(indata);
	elemsum->get_op<sum_elemwise_op>(0)->set_op_name("checker");

	*lb << elemsum;

	return lb;
}


layer* upsample(blob_base* data)
{
	layer *l = new layer(new data_args(data->channel(), 2, 2, 0, data->channel()));
	l->op(CACU_UN_POOLING, data);
	return l;
}



layer* linear(blob_base* data, int output_channel)
{
	layer *l = new layer(new data_args(output_channel, 0, 0, 0, data->length()));
	l->op<equalized_inner_product_op>(data);
	l->get_op<equalized_inner_product_op>(0)->set_use_wscale(true);
	l->get_op<equalized_inner_product_op>(0)->std_init_w();
	l->get_op<equalized_inner_product_op>(0)->set_is_use_bias(false);
	return l;
}


network* create_generator(int batch_size, int dim, int depth, phase_type phase_){

	blob *blob_= new blob(batch_size, dim * 8, 4, 4, 0, phase_);
	bin_blob *label_= new bin_blob(batch_size, 1, 1, 1, 0, phase_);
	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	in_datas->push_back(label_);

	network *generator = new network(in_datas);

	*generator << gblock_in_head(in_datas, dim * 8);

	int in_out[8] = {512, 512, 512, 256, 128, 64, 32, 16};

	for(int i = 0; i < depth - 1; ++i){
		if(i == 0){
			*generator << gblock_in(generator->output_blob(), label_, in_out[i]);
		}
		else{
			vector<scalex_op*> resGblocks = generator->get_ops<scalex_op>("resGblock_out");
			scalex_op *op = resGblocks[resGblocks.size() - 1];
			*generator << gblock_in(op->out_data<blob>(), label_, in_out[i]);
		}
	}

	return generator;
}

network* create_discriminator_linear(int batch_size, int dim, int depth, phase_type phase_){

	blob *blob_= new blob(batch_size, 3, std_size, std_size, 0, phase_);

	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	network *discriminator = new network(in_datas);

	int in_out[8] = {512, 512, 512, 256, 128, 64, 32, 16};
	for(int i = depth - 2; i >= 0; --i){
		if(i >= 1)
			*discriminator << dblock(discriminator->output_blob(), in_out[i], true);
		else
			*discriminator << dblock(discriminator->output_blob(), in_out[i], true);
	}

	layer *l = new layer(new data_args(512, 3, 1, 1, discriminator->output_blob()->channel()));
	l->op<equalized_convolution_op>(discriminator->output_blob())->op<leaky_relu_op>(new op_args(0.2));
	l->get_op<equalized_convolution_op>(0)->set_use_wscale(true);
	l->get_op<equalized_convolution_op>(0)->std_init_w();
	l->get_op<equalized_convolution_op>(0)->set_is_use_bias(false);
	*discriminator << l;

	*discriminator << linear(discriminator->output_blob(), 512);
	*discriminator << linear(discriminator->output_blob(), 1);


	return discriminator;
}

layer_block* create_mapping_function(int batch_size, int dim, int depth, phase_type phase_)
{

	blob *blob_= new blob(batch_size, 512, 1, 1, 0, phase_);

	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	layer_block *discriminator = new layer_block(in_datas);

	float_t lr = 0.01;

	layer *l = new layer(new data_args(512, 0, 0, 0, blob_->length()));
	l->op<equalized_inner_product_op>(blob_)->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l;

	layer *l1 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l1->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l1->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l1->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l1;

	layer *l1_1 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l1_1->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l1_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l1_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l1_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l1_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l1_1->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l1_1->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l1_1;

	layer *l2 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l2->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l2->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l2->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l2->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l2->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l2->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l2->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l2;

	layer *l2_1 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l2_1->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l2_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l2_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l2_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l2_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l2_1->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l2_1->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l2_1;

	layer *l3 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l3->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l3->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l3->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l3->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l3->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l3->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l3->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l3;

	layer *l3_1 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l3_1->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l3_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l3_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l3_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l3_1->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l3_1->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l3_1->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l3_1;

	layer *l4 = new layer(new data_args(512, 0, 0, 0, discriminator->get_oblob()->length()));
	l4->op<equalized_inner_product_op>(discriminator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.2));//->op(CACU_DROPOUT, new op_args(0.5));
	l4->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_lr_mul(lr);
	l4->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_use_wscale(true);
	l4->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->std_init_w();
	l4->get_op<equalized_inner_product_op>(0, CACU_E_INNERPRODUCT)->set_is_use_bias(false);
//	l4->get_op<equalized_inner_product_op>(0)->get_weight(0)->set_lr(lr);
//	l4->get_op<equalized_inner_product_op>(0)->get_weight(1)->set_lr(lr);

	*discriminator << l4;

	layer *split = new layer();
	split->op(CACU_SPLIT, discriminator->get_oblob(), new op_args(depth));

	*discriminator << split;

	int in_out[9] = {512, 512, 512, 512, 256, 128, 64, 32, 16};

	for(int i = 0; i < depth; ++i) {
		layer *ip1 = new layer(new data_args(in_out[i] * 2, 1, 1, 1, split->get_oblobs()->at(0)->length()));
		ip1->op<equalized_inner_product_op>(split->get_oblobs()->at(depth - i - 1));
		ip1->get_op<equalized_inner_product_op>(0)->set_use_wscale(true);
		ip1->get_op<equalized_inner_product_op>(0)->std_init_w();
		ip1->get_op<equalized_inner_product_op>(0)->set_is_use_bias(false);
		*discriminator << ip1;
	}

	return discriminator;
}


#endif /* EXAMPLE_CYCLE_FACE_MODELS_H_ */
