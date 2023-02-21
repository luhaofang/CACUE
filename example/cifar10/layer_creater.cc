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

#include "layer_creater.h"

using namespace cacu;

layer_block* conv_layer(blob_base* data, int output_channel,
	int kernel_size, int stride, int pad, op_name activation_op) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad,
		data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op); //->op(CACU_BATCH_NORMALIZE);
	l->get_op<convolution_op>(0, CACU_CONVOLUTION)->set_group(3);

	*lb << l;
	return lb;
}


layer_block* conv_layer_noActi(blob_base* data, int output_channel,
	int kernel_size, int stride, int pad, op_name activation_op) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad,
		data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE); //->op(CACU_BATCH_NORMALIZE);

	*lb << l;
	return lb;
}

layer_block* conv_layer_maxpooling(blob_base *data, int output_channel,
		int kernel_size, int stride, int pad, op_name activation_op) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad,
			data->channel()));
//	l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE);//->op<batch_normalize_op>()
	l->op<convolution_op>(data);
	layer *ml = new layer(new data_args(output_channel, 3, 2, 0, l->get_oblob()->channel()));
	ml->op(CACU_MAX_POOLING, l->get_oblob())->op(activation_op);

	*lb << l << ml;
	return lb;
}

layer_block* conv_layer_avgpooling(blob *data, int output_channel,
		int kernel_size, int stride, int pad, op_name activation_op) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad, data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(activation_op); //->op(CACU_BATCH_NORMALIZE);
	layer *al = new layer(new data_args(output_channel, 3, 2, 0, l->get_oblob()->channel()));
	al->op(CACU_AVERAGE_POOLING, (blob*) l->get_oblob());

	*lb << l << al;
	return lb;
}

layer_block* conv_layer_avgpooling_relu_first(blob_base* data,
		int output_channel, int kernel_size, int stride, int pad,
		op_name activation_op) {
	layer_block *lb = new layer_block();
	//LOG_DEBUG("%d", data);

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad,
			 data->channel()));
	l->op<convolution_op>(data)->op(activation_op);//->op<batch_normalize_op>()
	layer *al = new layer(new data_args(output_channel, 3, 2, 0, l->get_oblob()->channel()));
	al->op(CACU_AVERAGE_POOLING, l->get_oblob());

	*lb << l << al;
	return lb;
}

layer_block* conv_layer_maxpooling_relu_first(blob_base* data,
		int output_channel, int kernel_size, int stride, int pad,
		op_name activation_op) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel, kernel_size, stride, pad,
			 data->channel()));
	l->op(CACU_CONVOLUTION, data)->op(activation_op);
	layer *al = new layer(new data_args(output_channel, 2, 2, 0, l->get_oblob()->channel()));
	al->op(CACU_MAX_POOLING, l->get_oblob());

	*lb << l << al;
	return lb;
}

layer_block* fc_layer(blob_base* data, int output_channel, int kernel_size,
		int stride, int pad, op_name activation_op) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel,0,0,0,data->length()));
	l->op(CACU_INNERPRODUCT, data)->op(CACU_DROPOUT, new op_args(0.5)); //->op(CACU_BATCH_NORMALIZE)->op(activation_op);

	*lb << l;
	return lb;
}

layer_block* fc_layer_nodropout(blob_base* data, int output_channel,
		int kernel_size, int stride, int pad) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel,0,0,0,data->length()));
	l->op<inner_product_op>(data);//->op(CACU_BATCH_NORMALIZE);

	*lb << l;
	return lb;
}

layer_block* loss_layer(blob_base* data, blob_base* label, int output_channel) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel,0,0,0,data->length()));
	l->op(CACU_INNERPRODUCT, data)->op(CACU_SOFTMAX_LOSS, label);

	*lb << l;
	return lb;
}

layer_block* focal_loss_layer(blob_base* data, blob_base* label, int output_channel) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel,0,0,0,data->length()));
	l->op(CACU_INNERPRODUCT, data)->op<focal_softmax_with_loss_op>(label);
	l->get_op<focal_softmax_with_loss_op>(1)->set_lambda(1);

	*lb << l;
	return lb;
}

layer_block* loss_lm_layer(blob_base* data, blob_base* label, int output_channel) {
	layer_block *lb = new layer_block();

	blobs *bs = new blobs();
	bs->push_back(data);
	bs->push_back(label);
	layer *l = new layer(new data_args(output_channel,0,0,0,data->length()));
	l->op<inner_product_op>(bs)->op(CACU_SOFTMAX_LOSS, label);

	*lb << l;
	return lb;
}

layer_block* predict_layer(blob_base* data, int output_channel) {
	layer_block *lb = new layer_block();

	layer *l = new layer(new data_args(output_channel,0,0,0,data->length()));
	l->op(CACU_INNERPRODUCT, data)->op(CACU_SOFTMAX);

	*lb << l;
	return lb;
}

