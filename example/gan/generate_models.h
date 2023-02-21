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

#ifndef GENERATE_MODELS_H_
#define GENERATE_MODELS_H_

#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"
#include "../cifar10/layer_creater.h"

#include <time.h>

using namespace cacu;

layer_block* deconv_block(blob_base* data, int output_channel, int kernel_size,
		int stride = 1, int pad = 0, op_name activation_op = CACU_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_CONV_TRANS, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op); //
	l->get_op<batch_normalize_op>(1)->set_is_use_global_stats(false);
	//l->get_op<conv_transpose_op>(0)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* deconv_block_noactiv(blob_base* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_CONV_TRANS, data)->op(CACU_BATCH_NORMALIZE); //->op(CACU_BATCH_NORMALIZE);//
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* deconv_block_nobatchN(blob_base* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_CONV_TRANS, data)->op(activation_op); //->op(CACU_BATCH_NORMALIZE);//
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block(blob_base* data, int output_channel, int kernel_size,
		int stride = 1, int pad = 0, op_name activation_op = CACU_LEAKY_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op,
			new op_args(0.2)); //->op(CACU_DROPOUT, new op_args(0.5))
//	l->get_op<batch_normalize_op>(1)->set_is_use_global_stats(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_sn(blob_base* data, int output_channel, int kernel_size,
		int stride = 1, int pad = 0, op_name activation_op = CACU_LEAKY_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_SN_CONVOLUTION, data)->op(activation_op, new op_args(0.2))->op(CACU_DROPOUT, new op_args(0.5)); //->op(CACU_BATCH_NORMALIZE)
//	l->get_op<batch_normalize_op>(1)->set_is_use_global_stats(false);->op(CACU_DROPOUT, new op_args(0.5))
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_nobatchN(blob_base* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_LEAKY_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(activation_op); //
	//l->get_op<conv_transpose_op>(0)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_noactiv(blob_base* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_LEAKY_RELU) {
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(
			new data_args(output_channel, kernel_size, stride, pad,
					data->channel()));
	l->op(CACU_CONVOLUTION, data); //
	//l->get_op<conv_transpose_op>(0)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* sigmoid_loss(blob_base* data, blob_base *label) {

	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(1, 0, 0, 0, data->length()));
	l->op(CACU_INNERPRODUCT, data)->op(CACU_SIGMOID_LOSS, label);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* sigmoid_loss_sn(blob_base* data, blob_base *label) {

	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(1, 0, 0, 0, data->length()));
	l->op(CACU_SN_INNERPRODUCT, data)->op(CACU_SIGMOID_LOSS, label);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* hinge_loss_sn(blob_base* data, blob_base *label) {

	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(1, 0, 0, 0, data->length()));
	l->op(CACU_SN_INNERPRODUCT, data)->op(CACU_HINGE_LOSS, label);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* create_generator(blob *blob_, phase_type phase_) {

	layer_block *generator = new layer_block();

	layer_block *deconv1_2 = deconv_block(blob_, 2048, 4, 1, 0);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv2_2 = deconv_block(deconv1_2->get_oblob(), 1024, 4, 2,
			1);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv3_2 = deconv_block(deconv2_2->get_oblob(), 512, 4, 2, 1);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv4_2 = deconv_block(deconv3_2->get_oblob(), 256, 4, 2, 1);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv5_2 = deconv_block(deconv4_2->get_oblob(), 128, 4, 2, 1);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv6_2 = deconv_block_nobatchN(deconv5_2->get_oblob(), 3, 4,
			2, 1, CACU_TANH);
	deconv6_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv6_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv6_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	*generator << deconv1_2 << deconv2_2 << deconv3_2 << deconv4_2 << deconv5_2
			<< deconv6_2;

	return generator;
}

layer_block* create_discriminator(blob *blob_, bin_blob *label_,
		phase_type phase_) {

	layer_block *discriminator = new layer_block();

	layer_block *conv1_1 = conv_block(blob_, 128, 4, 2, 1);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv2_1 = conv_block(conv1_1->get_oblob(), 256, 4, 2, 1);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv3_1 = conv_block(conv2_1->get_oblob(), 512, 4, 2, 1);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv4_1 = conv_block(conv3_1->get_oblob(), 1024, 4, 2, 1);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv5_1 = conv_block(conv4_1->get_oblob(), 2048, 4, 2, 1);
	conv5_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv5_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv5_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	*discriminator << conv1_1 << conv2_1 << conv3_1 << conv4_1 << conv5_1;
	//*/
	if (phase_ == train) {

		layer_block *loss_ = sigmoid_loss(conv5_1->get_oblob(), label_);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(
				gaussian, 0.01);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(
				constant);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->get_weight(
				1)->set_decay(0);

		*discriminator << loss_;
	} else {
		layer_block *predict_ = predict_layer(conv5_1->get_oblob(), 2);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(
				gaussian, 0.01);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(
				constant);

		*discriminator << predict_;
	}
	//*/
	return discriminator;
}

layer_block* create_generator_96(blob *blob_, phase_type phase_) {

	layer_block *generator = new layer_block();

	layer_block *deconv1_2 = deconv_block(blob_, 2048, 3, 1, 0);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv2_2 = deconv_block(deconv1_2->get_oblob(), 1024, 4, 2,
			1);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv3_2 = deconv_block(deconv2_2->get_oblob(), 512, 4, 2, 1);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv4_2 = deconv_block(deconv3_2->get_oblob(), 256, 4, 2, 1);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv5_2 = deconv_block(deconv4_2->get_oblob(), 128, 4, 2, 1);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv6_2 = deconv_block_nobatchN(deconv5_2->get_oblob(), 3, 4,
			2, 1, CACU_TANH);
	deconv6_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv6_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv6_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	*generator << deconv1_2 << deconv2_2 << deconv3_2 << deconv4_2 << deconv5_2
			<< deconv6_2;

	return generator;
}

layer_block* create_discriminator_96(blob *blob_, bin_blob *label_,
		phase_type phase_) {

	layer_block *discriminator = new layer_block();

	layer_block *conv1_1 = conv_block(blob_, 128, 4, 2, 1);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv2_1 = conv_block(conv1_1->get_oblob(), 256, 4, 2, 1);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv3_1 = conv_block(conv2_1->get_oblob(), 512, 4, 2, 1);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv4_1 = conv_block(conv3_1->get_oblob(), 1024, 4, 2, 1);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv5_1 = conv_block(conv4_1->get_oblob(), 2048, 4, 2, 1);
	conv5_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv5_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv5_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	*discriminator << conv1_1 << conv2_1 << conv3_1 << conv4_1 << conv5_1;
	//*/
	if (phase_ == train) {

		layer_block *loss_ = sigmoid_loss(conv5_1->get_oblob(), label_);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(
				gaussian, 0.01);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(
				constant);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->get_weight(
				1)->set_decay(0);

		*discriminator << loss_;
	} else {
		layer_block *predict_ = predict_layer(conv5_1->get_oblob(), 2);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(
				gaussian, 0.01);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(
				constant);

		*discriminator << predict_;
	}
	//*/
	return discriminator;
}

layer_block* create_generator_64(blob *blob_, int dim_, phase_type phase_) {

	layer_block *generator = new layer_block();

	layer_block *deconv1_2 = deconv_block_noactiv(blob_, dim_ * 8, 4, 1, 0);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv1_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv2_2 = deconv_block(deconv1_2->get_oblob(), dim_ * 4, 5,
			2, 2);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv2_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv3_2 = deconv_block(deconv2_2->get_oblob(), dim_ * 2, 5,
			2, 2);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv3_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv4_2 = deconv_block(deconv3_2->get_oblob(), dim_, 5, 2,
			2);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv4_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *deconv5_2 = deconv_block_nobatchN(deconv4_2->get_oblob(), 3, 5,
			2, 2, CACU_TANH);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->set_weight_init_type(
			gaussian, 0.01);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->set_bias_init_type(
			constant);
	deconv5_2->layers(0)->get_op<conv_transpose_op>(0)->get_weight(1)->set_decay(
			0);

	*generator << deconv1_2 << deconv2_2 << deconv3_2 << deconv4_2 << deconv5_2;

	return generator;
}

layer_block* create_discriminator_64(blob *blob_, int dim_, bin_blob *label_,
		phase_type phase_) {

	layer_block *discriminator = new layer_block();

	layer_block *conv1_1 = conv_block(blob_, dim_, 5, 2, 2);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv1_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv2_1 = conv_block(conv1_1->get_oblob(), dim_ * 2, 5, 2, 2);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv2_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv3_1 = conv_block(conv2_1->get_oblob(), dim_ * 4, 5, 2, 2);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv3_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	layer_block *conv4_1 = conv_block(conv3_1->get_oblob(), dim_ * 8, 5, 2, 2);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_weight_init_type(
			gaussian, 0.01);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_bias_init_type(
			constant);
	conv4_1->layers(0)->get_op<convolution_op>(0, CACU_CONVOLUTION)->get_weight(
			1)->set_decay(0);

	*discriminator << conv1_1 << conv2_1 << conv3_1 << conv4_1;
	//*/
	if (phase_ == train) {

		layer_block *loss_ = sigmoid_loss(conv4_1->get_oblob(), label_);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(
				gaussian, 0.01);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(
				constant);
		loss_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->get_weight(
				1)->set_decay(0);

		*discriminator << loss_;
	} else {
		layer_block *predict_ = predict_layer(conv4_1->get_oblob(), 2);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_weight_init_type(
				gaussian, 0.01);
		predict_->layers(0)->get_op<inner_product_op>(0, CACU_INNERPRODUCT)->set_bias_init_type(
				constant);

		*discriminator << predict_;
	}
	//*/
	return discriminator;
}

layer_block* create_discriminator_sn_64(blob *blob_, int dim_, bin_blob *label_,
		phase_type phase_) {

	layer_block *discriminator = new layer_block();

	layer_block *conv1_1 = conv_block_sn(blob_, dim_, 5, 2, 2);
	conv1_1->layers(0)->get_op<sn_convolution_op>(0)->set_weight_init_type(
			gaussian, 0.1);
	conv1_1->layers(0)->get_op<sn_convolution_op>(0)->set_bias_init_type(
			constant);
	conv1_1->layers(0)->get_op<sn_convolution_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *conv2_1 = conv_block_sn(conv1_1->get_oblob(), dim_ * 2, 5, 2,
			2);
	conv2_1->layers(0)->get_op<sn_convolution_op>(0)->set_weight_init_type(
			gaussian, 0.1);
	conv2_1->layers(0)->get_op<sn_convolution_op>(0)->set_bias_init_type(
			constant);
	conv2_1->layers(0)->get_op<sn_convolution_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *conv3_1 = conv_block_sn(conv2_1->get_oblob(), dim_ * 4, 5, 2,
			2);
	conv3_1->layers(0)->get_op<sn_convolution_op>(0)->set_weight_init_type(
			gaussian, 0.1);
	conv3_1->layers(0)->get_op<sn_convolution_op>(0)->set_bias_init_type(
			constant);
	conv3_1->layers(0)->get_op<sn_convolution_op>(0)->get_weight(1)->set_decay(
			0);

	layer_block *conv4_1 = conv_block_sn(conv3_1->get_oblob(), dim_ * 8, 5, 2,
			2);
	conv4_1->layers(0)->get_op<sn_convolution_op>(0)->set_weight_init_type(
			gaussian, 0.1);
	conv4_1->layers(0)->get_op<sn_convolution_op>(0)->set_bias_init_type(
			constant);
	conv4_1->layers(0)->get_op<sn_convolution_op>(0)->get_weight(1)->set_decay(
			0);

	*discriminator << conv1_1 << conv2_1 << conv3_1 << conv4_1;
	//*/
	if (phase_ == train) {

		layer_block *loss_ = sigmoid_loss_sn(conv4_1->get_oblob(), label_);
		loss_->layers(0)->get_op<sn_inner_product_op>(0)->set_weight_init_type(
				gaussian, 0.1);
		loss_->layers(0)->get_op<sn_inner_product_op>(0)->set_bias_init_type(
				constant);
		loss_->layers(0)->get_op<sn_inner_product_op>(0)->get_weight(1)->set_decay(
				0);

		*discriminator << loss_;
	} else {
		layer_block *predict_ = predict_layer(conv4_1->get_oblob(), 2);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_weight_init_type(
				gaussian, 0.1);
		predict_->layers(0)->get_op<inner_product_op>(0)->set_bias_init_type(
				constant);

		*discriminator << predict_;
	}
	//*/
	return discriminator;
}

#endif
