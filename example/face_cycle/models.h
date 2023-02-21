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

#ifndef EXAMPLE_CYCLE_FACE_MODELS_H_
#define EXAMPLE_CYCLE_FACE_MODELS_H_


#include "../cifar10/layer_creater.h"

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"

#include <time.h>

using namespace cacu;

#define CLASS_NUM 526

layer_block* conv_block(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	layer *l1 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l1->op<convolution_op>(data)->op<relu_op>();
	l1->get_op<convolution_op>(0)->set_weight_init_type(xavier);
	l1->get_op<convolution_op>(0)->set_is_use_bias(usebias);

	*lb << l1;
	return lb;
}

layer_block* conv_block_with_pooling(blob_base* data,int output_channel, int kernel_size, int stride,int pad, bool usebias = true)
{
	layer_block *lb = new layer_block();
	layer *l1 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l1->op<convolution_op>(data)->op<relu_op>();
	l1->get_op<convolution_op>(0)->set_weight_init_type(xavier);
	l1->get_op<convolution_op>(0)->set_is_use_bias(usebias);

	layer *pooling = new layer(new data_args(l1->get_oblob()->channel(), 3, 2, 0, l1->get_oblob()->channel()));
	pooling->op<max_pooling_op>(l1->get_oblob());

	*lb << l1 << pooling;

	return lb;
}

layer_block* deconv_block_noactiv_cgan(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_LEAKY_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONV_TRANS, data);
	l->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_weight_init_type(xavier);
	l->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_res_block_dis(blob_base* data, int output_channel, int kernel_size, int stride, int pad, bool usebias = false, bool isnorm = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *split = new layer();
	split->op(CACU_SPLIT, data, new op_args(2));

	layer *ls = new layer(new data_args(output_channel, 1, 1, 0, data->channel()));
	ls->op(CACU_SN_CONVOLUTION, split->get_oblobs()->at(0));
	ls->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier);
	ls->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *ds_x = new layer(new data_args(ls->get_oblob()->channel(), 2, 2, 0, ls->get_oblob()->channel()));
	ds_x->op(CACU_AVERAGE_POOLING, ls->get_oblob());

	layer *l1 = new layer(new data_args(data->channel(), kernel_size, stride, pad, data->channel()));
	l1->op(CACU_LEAKY_RELU, split->get_oblobs()->at(1), new op_args(0.02))->op(CACU_SN_CONVOLUTION)->op(CACU_LEAKY_RELU, new op_args(0.02));
	l1->get_op<sn_convolution_op>(1, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	l1->get_op<sn_convolution_op>(1, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l2->op(CACU_SN_CONVOLUTION, l1->get_oblob());
	l2->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	l2->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *ds_r = new layer(new data_args(l2->get_oblob()->channel(), 2, 2, 0, l2->get_oblob()->channel()));
	ds_r->op(CACU_AVERAGE_POOLING, l2->get_oblob());

	blobs *sum_data = new blobs();
	sum_data->push_back(ds_x->get_oblob());
	sum_data->push_back(ds_r->get_oblob());

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);
	clock_t end = clock();
	*lb << split << ls << ds_x << l1 << l2 << ds_r << elemsum;
	return lb;
}

layer_block* conv_res_block_nods_dis(blob_base* data, int output_channel, int kernel_size, int stride, int pad, bool usebias = false, bool isnorm = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *split = new layer();
	split->op(CACU_SPLIT, data, new op_args(2));

	layer *l1 = new layer(new data_args(data->channel(), kernel_size, stride, pad, data->channel()));
	l1->op(CACU_LEAKY_RELU, split->get_oblobs()->at(1), new op_args(0.02))->op(CACU_SN_CONVOLUTION)->op(CACU_LEAKY_RELU, new op_args(0.02));
	l1->get_op<sn_convolution_op>(1, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	l1->get_op<sn_convolution_op>(1, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l2->op(CACU_SN_CONVOLUTION, l1->get_oblob());
	l2->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	l2->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	blobs *sum_data = new blobs();
	sum_data->push_back(split->get_oblobs()->at(0));
	sum_data->push_back(l2->get_oblob());

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);
	clock_t end = clock();
	*lb << split << l1 << l2 << elemsum;
	return lb;
}

layer_block* optimized_block(blob_base* data, int output_channel, int kernel_size, int stride,int pad, bool usebias = false, bool isnorm = false)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *split = new layer();
	split->op(CACU_SPLIT, data, new op_args(2));

	layer *ds_x = new layer(new data_args(split->get_oblobs()->at(0)->channel(), 2, 2, 0, split->get_oblobs()->at(0)->channel()));
	ds_x->op(CACU_AVERAGE_POOLING, split->get_oblobs()->at(0));

	layer *ls = new layer(new data_args(output_channel, 1, 1, 0, ds_x->get_oblob()->channel()));
	ls->op(CACU_SN_CONVOLUTION, ds_x->get_oblob());
	ls->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier);
	ls->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_SN_CONVOLUTION, split->get_oblobs()->at(1))->op(CACU_LEAKY_RELU, new op_args(0.02));
	l->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	l->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *l2 = new layer(new data_args(output_channel, kernel_size, stride, pad, l->get_oblob()->channel()));
	l2->op(CACU_SN_CONVOLUTION, l->get_oblob());
	l2->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	l2->get_op<sn_convolution_op>(0, CACU_SN_CONVOLUTION)->set_is_use_bias(usebias);

	layer *ds_r = new layer(new data_args(l2->get_oblob()->channel(), 2, 2, 0, l2->get_oblob()->channel()));
	ds_r->op(CACU_AVERAGE_POOLING, l2->get_oblob());

	blobs *sum_data = new blobs();
	sum_data->push_back(ls->get_oblob());
	sum_data->push_back(ds_r->get_oblob());

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);
	clock_t end = clock();
	*lb << split << ds_x << ls << l << l2 << ds_r << elemsum;
	return lb;
}

layer* upsample(blob_base* data)
{
	layer *l = new layer(new data_args(data->channel(), 2, 2, 0, data->channel()));
	l->op(CACU_UN_POOLING, data);
//	l->op(CACU_CONV_TRANS, data);
//	l->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_weight_init_type(constant, 1);
//	l->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_is_use_bias(true);
//	l->get_op<conv_transpose_op>(0, CACU_CONV_TRANS)->set_group(data->channel());
	return l;
}

layer_block* deconv_res_block_gen(blob_base* data, blob_base* labels, int output_channel, int kernel_size, int stride,int pad, bool usebias = false)
{

	layer_block *lb = new layer_block();
	clock_t start = clock();

	layer *split = new layer();
	split->op(CACU_SPLIT, data, new op_args(2));

	layer *us_x = upsample(split->get_oblobs()->at(0));

	layer *lc_x = new layer(new data_args(output_channel, 1, 1, 0, data->channel()));
	lc_x->op(CACU_CONVOLUTION, us_x->get_oblob());
	lc_x->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(xavier);
	lc_x->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);

	blobs *datas = new blobs();
	datas->push_back(split->get_oblobs()->at(1));
	datas->push_back(labels);

	layer *l = new layer(new data_args(data->channel(), 0, 0, 0, data->channel()));
	l->op(CACU_CONDITIONAL_BN, datas, new op_args((float_t)CLASS_NUM))->op(CACU_LEAKY_RELU, new op_args(0.02));
	l->get_op<conditional_bn_op>(0, CACU_CONDITIONAL_BN)->set_is_use_global_stats(false);

	layer *us_r = upsample(l->get_oblob());

	layer *lc_r1 = new layer(new data_args(data->channel(), kernel_size, stride, pad, data->channel()));
	lc_r1->op(CACU_CONVOLUTION, us_r->get_oblob())->op(CACU_CONDITIONAL_BN, labels, new op_args((float_t)CLASS_NUM))->op(CACU_LEAKY_RELU, new op_args(0.02));
	lc_r1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	lc_r1->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);
	lc_r1->get_op<conditional_bn_op>(1, CACU_CONDITIONAL_BN)->set_is_use_global_stats(false);

	layer *lc_r2 = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	lc_r2->op(CACU_CONVOLUTION, lc_r1->get_oblob());
	lc_r2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(xavier, sqrt(2));
	lc_r2->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(usebias);

	blobs *sum_data = new blobs();
	sum_data->push_back(lc_x->get_oblob());
	sum_data->push_back(lc_r2->get_oblob());

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);
	clock_t end = clock();
	*lb << split << us_x << lc_x << l << us_r << lc_r1 << lc_r2 << elemsum;
	return lb;
}

layer* sn_linear(blob_base* data, int output_channel)
{
	layer *l = new layer(new data_args(output_channel, 0, 0, 0, data->length()));
	l->op(CACU_SN_INNERPRODUCT, data);
	l->get_op<sn_inner_product_op>(0, CACU_SN_INNERPRODUCT)->set_weight_init_type(xavier);
	l->get_op<sn_inner_product_op>(0, CACU_SN_INNERPRODUCT)->set_is_use_bias(false);
	return l;
}


layer_block* create_generator_cycleface(int batch_size, int dim, phase_type phase_)
{
	blob *blob_= new blob(batch_size, 128, 1, 1, 0, phase_);
	bin_blob *labels = new bin_blob(batch_size, 1, 1, 1, 0, phase_);
	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	in_datas->push_back(labels);
	layer_block *generator = new layer_block(in_datas);

	*generator << deconv_block_noactiv_cgan(blob_, dim * 16, 4, 1, 0);
	*generator << deconv_res_block_gen(generator->get_oblob(), labels, dim * 16, 3, 1, 1);
	*generator << deconv_res_block_gen(generator->get_oblob(), labels, dim * 8, 3, 1, 1);
	*generator << deconv_res_block_gen(generator->get_oblob(), labels, dim * 4, 3, 1, 1);
	*generator << deconv_res_block_gen(generator->get_oblob(), labels, dim * 2, 3, 1, 1);
	*generator << deconv_res_block_gen(generator->get_oblob(), labels, dim, 3, 1, 1);

	layer *bn = new layer(new data_args(generator->get_oblob()->channel(), 0, 0, 0, 0));
	bn->op(CACU_BATCH_NORMALIZE, generator->get_oblob())->op(CACU_LEAKY_RELU, new op_args(0.02));
	bn->get_op<batch_normalize_op>(0, CACU_BATCH_NORMALIZE)->set_is_use_global_stats(false);

	*generator << bn;

	layer *conv = new layer(new data_args(3, 3, 1, 1, dim));
	conv->op(CACU_CONVOLUTION, generator->get_oblob())->op(CACU_TANH);
	conv->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(xavier);
	conv->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_is_use_bias(false);

	*generator << conv;

	return generator;
}

layer_block* create_discriminator_regression_cycleface(int batch_size, phase_type phase_)
{

	blob *blob_= new blob(batch_size, 3, 128, 128, 0, phase_);

	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	layer_block *discriminator = new layer_block(in_datas);

	*discriminator << conv_block_with_pooling(blob_, 32, 3, 1, 0);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 64, 3, 1, 0);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 64, 3, 1, 0);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 128, 3, 1, 0);
	*discriminator << conv_block(discriminator->get_oblob(), 256, 3, 1, 0);

	layer *split = new layer();
	split->op<split_op>(discriminator->get_oblob(), new op_args(2));

	*discriminator << split;

	layer *ip1 = new layer(new data_args(128, 1, 1, 1, split->get_oblobs()->at(0)->length()));
	ip1->op<inner_product_op>(split->get_oblobs()->at(0))->op<tanh_op>();
	ip1->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	ip1->get_op<inner_product_op>(0)->set_is_use_bias(true);

	*discriminator << ip1;

	layer *ip2 = new layer(new data_args(256, 1, 1, 1, split->get_oblobs()->at(1)->length()));
	ip2->op<inner_product_op>(split->get_oblobs()->at(1))->op<relu_op>()->op<dropout_op>(new op_args(0.5));
	ip2->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	ip2->get_op<inner_product_op>(0)->set_is_use_bias(true);

	*discriminator << ip2;

	layer *ip3 = new layer(new data_args(10, 1, 1, 1, ip2->get_oblob()->length()));
	ip3->op<inner_product_op>(ip2->get_oblob());
	ip3->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	ip3->get_op<inner_product_op>(0)->set_is_use_bias(true);

	*discriminator << ip3;

	return discriminator;
}



layer_block* create_discriminator_68regression_cycleface(int batch_size, phase_type phase_)
{

	blob *blob_= new blob(batch_size, 3, 128, 128, 0, phase_);

	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	layer_block *discriminator = new layer_block(in_datas);

	*discriminator << conv_block_with_pooling(blob_, 32, 3, 1, 0);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 64, 3, 1, 0);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 128, 3, 1, 0);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 256, 3, 1, 0);
	*discriminator << conv_block(discriminator->get_oblob(), 512, 3, 1, 0);

	layer *ip2 = new layer(new data_args(512, 1, 1, 1, discriminator->get_oblob()->length()));
	ip2->op<inner_product_op>(discriminator->get_oblob())->op<relu_op>()->op<dropout_op>(new op_args(0.5));
	ip2->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	ip2->get_op<inner_product_op>(0)->set_is_use_bias(true);

	*discriminator << ip2;

	layer *ip3 = new layer(new data_args(136, 1, 1, 1, ip2->get_oblob()->length()));
	ip3->op<inner_product_op>(ip2->get_oblob());
	ip3->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	ip3->get_op<inner_product_op>(0)->set_is_use_bias(true);

	*discriminator << ip3;

	return discriminator;
}


layer_block* create_discriminator_random_cycleface(int batch_size, phase_type phase_)
{

	blob *blob_= new blob(batch_size, 3, 128, 128, 0, phase_);

	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	layer_block *discriminator = new layer_block(in_datas);

	*discriminator << conv_block_with_pooling(blob_, 64, 3, 1, 1);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 128, 3, 1, 1);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 256, 3, 1, 1);
	*discriminator << conv_block_with_pooling(discriminator->get_oblob(), 512, 3, 1, 1);
	*discriminator << conv_block(discriminator->get_oblob(), 1024, 3, 1, 1);

	layer *ip1 = new layer(new data_args(128, 1, 1, 1, discriminator->get_oblob()->length()));
	ip1->op<inner_product_op>(discriminator->get_oblob())->op<tanh_op>();
	ip1->get_op<inner_product_op>(0)->set_weight_init_type(xavier);
	ip1->get_op<inner_product_op>(0)->set_is_use_bias(true);

	*discriminator << ip1;

	return discriminator;
}

layer_block* create_discriminator_cycleface(int batch_size, int dim, phase_type phase_)
{
	blob *blob_= new blob(batch_size, 3, 128, 128, 0, phase_);
	bin_blob *label_ = new bin_blob(batch_size, 1, 1, 1, 0, phase_);

	blobs *in_datas = new blobs();
	in_datas->push_back(blob_);
	in_datas->push_back(label_);
	layer_block *discriminator = new layer_block(in_datas);

	*discriminator << optimized_block(blob_, dim, 3, 1, 1);

	*discriminator << conv_res_block_dis(discriminator->get_oblob(), dim * 2, 3, 1, 1);
	*discriminator << conv_res_block_dis(discriminator->get_oblob(), dim * 4, 3, 1, 1);
	*discriminator << conv_res_block_dis(discriminator->get_oblob(), dim * 8, 3, 1, 1);
	*discriminator << conv_res_block_dis(discriminator->get_oblob(), dim * 16, 3, 1, 1);
	*discriminator << conv_res_block_nods_dis(discriminator->get_oblob(), dim * 16, 3, 1, 1);

	layer *gp = new layer(new data_args(discriminator->get_oblob()->channel(), discriminator->get_oblob()->width(),
			1, 0, discriminator->get_oblob()->channel()));
	gp->op(CACU_LEAKY_RELU, discriminator->get_oblob(), new op_args(0.02))->op(CACU_AVERAGE_POOLING)->op(CACU_MATH_SCALEX, new op_args((float_t)discriminator->get_oblob()->channel_length()));
	*discriminator << gp;

//	layer *sn_linear_ = sn_linear(discriminator->get_oblob(), 1);
//	*discriminator << sn_linear_;

	layer *split = new layer();
	split->op(CACU_SPLIT, discriminator->get_oblob(), new op_args(2));
	*discriminator << split;

	layer *sn_linear_ = sn_linear(split->get_oblobs()->at(0), 1);
	*discriminator << sn_linear_;

	blobs *projection_blobs = new blobs();
	projection_blobs->push_back(split->get_oblobs()->at(1));
	projection_blobs->push_back(label_);

//	layer *projection = new layer(new data_args(1, 0, 0, 0, split->get_oblobs()->at(1)->length()));
//	projection->op(CACU_IP_PROJECTION, projection_blobs, new op_args(CLASS_NUM));
//	projection->get_op<ip_projection_op>(0, CACU_IP_PROJECTION)->set_weight_init_type(xavier);
//	projection->get_op<ip_projection_op>(0, CACU_IP_PROJECTION)->set_is_use_bias(false);
//	projection->get_op<ip_projection_op>(0, CACU_IP_PROJECTION)->set_is_norm_weights(true);

	layer *projection = new layer(new data_args(CLASS_NUM, 0, 0, 0, split->get_oblobs()->at(1)->length()));
	projection->op(CACU_SN_INNERPRODUCT, split->get_oblobs()->at(1))->op(CACU_CONDITIONAL_PROJECTION, label_);
	projection->get_op<sn_inner_product_op>(0, CACU_SN_INNERPRODUCT)->set_weight_init_type(xavier);
	projection->get_op<sn_inner_product_op>(0, CACU_SN_INNERPRODUCT)->set_is_use_bias(false);

	*discriminator << projection;

	blobs *sum_data = new blobs();
	sum_data->push_back(sn_linear_->get_oblob());
	sum_data->push_back(projection->get_oblob());

	layer *elemsum = new layer();
	elemsum->op(CACU_SUM_ELEMWISE,sum_data);//->op(CACU_NORM);
	*discriminator << elemsum;

	return discriminator;
}




#endif /* EXAMPLE_CYCLE_FACE_MODELS_H_ */
