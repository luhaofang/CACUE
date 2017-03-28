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

#pragma once


namespace mycnn{

	layer_block* conv_layer_maxpooling(blob* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
		l->op(CACU_CONVOLUTION, data);
		layer *ml = new layer(output_channel, 3, 2);
		ml->op(CACU_MAX_POOLING, (blob*)l->get_oblob())->op(activation_op);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l << ml;
		return lb;
	}

	layer_block* conv_layer_avgpooling(blob* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
		l->op(CACU_CONVOLUTION, data);
		layer *al = new layer(output_channel, 3, 2);
		al->op(CACU_AVERAGE_POOLING, (blob*)l->get_oblob())->op(activation_op);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l << al;
		return lb;
	}

	layer_block* conv_layer_avgpooling_relu_first(blob* data,int output_channel, int kernel_size, int stride = 1, int pad = 0,op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
		l->op(CACU_CONVOLUTION, data)->op(activation_op);
		layer *al = new layer(output_channel, 3, 2);
		al->op(CACU_AVERAGE_POOLING, (blob*)l->get_oblob());
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l << al;
		return lb;
	}

	layer_block* conv_layer_nopooling(blob* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel, kernel_size, stride, pad, data->height(), data->channel());
		l->op(CACU_CONVOLUTION, data)->op(CACU_BATCH_NORMALIZE)->op(activation_op);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l;
		return lb;
	}

	layer_block* fc_layer(blob* data, int output_channel, int kernel_size = 0, int stride = 0, int pad = 0, op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel);
		l->op(CACU_INNERPRODUCT, data)->op(CACU_DROPOUT)->op(activation_op);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l;
		return lb;
	}

	layer_block* fc_layer_nodropout(blob* data, int output_channel, int kernel_size = 0, int stride = 0, int pad = 0)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel);
		l->op(CACU_INNERPRODUCT, data);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l;
		return lb;
	}

	layer_block* loss_layer(blob* data, bin_blob* label, int output_channel)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel);
		l->op(CACU_INNERPRODUCT, data)->op(CACU_SOFTMAX_LOSS, label);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l;
		return lb;
	}

	layer_block* predict_layer(blob* data, int output_channel)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel);
		l->op(CACU_INNERPRODUCT, data)->op(CACU_SOFTMAX);
		clock_t end = clock();
		LOG_INFO("time cost :%d", (end - start));
		*lb << l;
		return lb;
	}

};
