/*
 * generate_models_cifar.h
 *
 *  Created on: May 30, 2018
 *      Author: haofang
 */

#ifndef GENERATE_MODELS_CIFAR_H_
#define GENERATE_MODELS_CIFAR_H_


#include "../../cacu/cacu.h"
#include "../../cacu/framework/network.h"
#include "../../cacu/framework/layer.h"
#include "../cifar10/layer_creater.h"

#include <time.h>

using namespace cacu;


layer_block* deconv_block(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_DECONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op); //
	//l->get_op<deconvolution_op>(0)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* deconv_block_noactiv(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_DECONVOLUTION, data)->op(CACU_BATCH_NORMALIZE);//->op(CACU_BATCH_NORMALIZE);//
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* deconv_block_nobatchN(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_DECONVOLUTION, data)->op(activation_op);//->op(CACU_BATCH_NORMALIZE);//
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_LEAKY_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op,new op_args(0.01)); //
	//l->get_op<deconvolution_op>(0)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_nobatchN(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_LEAKY_RELU)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(activation_op,new op_args(0.01)); //
	//l->get_op<deconvolution_op>(0)->set_is_use_bias(false);
	clock_t end = clock();
	*lb << l;
	return lb;
}

layer_block* conv_block_noactiv(blob_base* data, int output_channel, int kernel_size, int stride = 1, int pad = 0)
{
	layer_block *lb = new layer_block();
	clock_t start = clock();
	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data); //
	//l->get_op<deconvolution_op>(0)->set_is_use_bias(false);
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

layer_block* create_generator_32(blob *blob_, phase_type phase_)
{

	layer_block *generator = new layer_block();

	layer_block *deconv1 = deconv_block_noactiv(blob_, 512, 4, 1, 0);
	deconv1->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_weight_init_type(constant,0.01);
	deconv1->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_bias_init_type(constant);
	deconv1->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->get_weight(1)->set_decay(0);

	layer_block *deconv1_2 = deconv_block(deconv1->get_oblob(), 256, 4, 2, 1);
	deconv1_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_weight_init_type(constant,0.01);
	deconv1_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_bias_init_type(constant);
	deconv1_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->get_weight(1)->set_decay(0);

	layer_block *deconv2_2 = deconv_block(deconv1_2->get_oblob(), 128, 4, 2, 1);
	deconv2_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_weight_init_type(constant,0.01);
	deconv2_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_bias_init_type(constant);
	deconv2_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->get_weight(1)->set_decay(0);

	layer_block *deconv3_2 = deconv_block_nobatchN(deconv2_2->get_oblob(), 3, 4, 2, 1, CACU_TANH);
	deconv3_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_weight_init_type(constant,0.01);
	deconv3_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->set_bias_init_type(constant);
	deconv3_2->layers(0)->get_op<deconvolution_op>(0,CACU_DECONVOLUTION)->get_weight(1)->set_decay(0);

	*generator << deconv1 << deconv1_2 << deconv2_2 << deconv3_2;

	return generator;
}

layer_block* create_discriminator_32(blob *blob_, bin_blob *label_, phase_type phase_)
{

	layer_block *net = new layer_block();

	layer_block *conv2 = conv_block(blob_, 128, 4, 2, 1);
	conv2->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->set_weight_init_type(constant,0.01);
	conv2->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->set_bias_init_type(constant);
	conv2->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->get_weight(1)->set_decay(0);

	layer_block *conv3 = conv_block(conv2->get_oblob(), 256, 4, 2, 1);
	conv3->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->set_weight_init_type(constant,0.01);
	conv3->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->set_bias_init_type(constant);
	conv3->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->get_weight(1)->set_decay(0);

	layer_block *conv4 = conv_block(conv3->get_oblob(), 512, 4, 2, 1);
	conv4->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->set_weight_init_type(constant,0.01);
	conv4->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->set_bias_init_type(constant);
	conv4->layers(0)->get_op<convolution_op>(0,CACU_CONVOLUTION)->get_weight(1)->set_decay(0);

	*net << conv2 << conv3 << conv4;

	if(phase_ == train){

		layer_block *loss_ = sigmoid_loss(conv4->get_oblob(), label_);
		loss_->layers(0)->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->set_weight_init_type(constant,0.01);
		loss_->layers(0)->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->set_bias_init_type(constant);
		loss_->layers(0)->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->get_weight(1)->set_decay(0);

		*net << loss_;
	}
	else
	{
		layer_block *predict_ = predict_layer(conv4->get_oblob(), 2);
		predict_->layers(0)->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->set_weight_init_type(constant,0.01);
		predict_->layers(0)->get_op<inner_product_op>(0,CACU_INNERPRODUCT)->set_bias_init_type(constant);

		*net << predict_;
	}


	return net;
}



#endif /* GENERATE_MODELS_CIFAR_H_ */
