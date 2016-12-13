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
		layer *l = new layer(output_channel, kernel_size, stride, pad);
		l->op(CACU_CONVOLUTION,data)->op(activation_op);
		layer *ml = new layer(output_channel, 3, 2);
		ml->op(CACU_MAX_POOLING, (blob*)l->get_oblob());
		clock_t end = clock();
		LOG_INFO("time cost :%d\n", (end - start) / 1000);
		*lb << l << ml;
		return lb;
	}

	layer_block* conv_layer_nopooling(blob* data, int output_channel, int kernel_size, int stride = 1, int pad = 0, op_name activation_op = CACU_RELU)
	{
		layer_block *lb = new layer_block();
		clock_t start = clock();
		layer *l = new layer(output_channel, kernel_size, stride, pad);
		l->op(CACU_CONVOLUTION, data)->op(activation_op);
		clock_t end = clock();
		LOG_INFO("time cost :%d\n", (end - start) / 1000);
		*lb << l;
		return lb;
	}

};
