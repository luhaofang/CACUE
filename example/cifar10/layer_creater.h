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

#ifndef LAYER_CREATER_H_
#define LAYER_CREATER_H_

#include "../../cacu/framework/layer.h"
#include "../../cacu/framework/layer_block.h"
#include "../../cacu/framework/blob_base.h"

#include "../../cacu/ops/ops_definition.h"
#include "../../cacu/ops/operators.h"

using namespace cacu;

layer_block* conv_layer_maxpooling(blob_base* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_RELU);

layer_block* conv_layer_avgpooling(blob* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_RELU);

layer_block* conv_layer_avgpooling_relu_first(blob_base* data,
		int output_channel, int kernel_size, int stride = 1, int pad = 0,
		op_name activation_op = CACU_RELU);

layer_block* conv_layer_maxpooling_relu_first(blob_base* data,
		int output_channel, int kernel_size, int stride = 1, int pad = 0,
		op_name activation_op = CACU_RELU);

layer_block* conv_layer_nopooling_norelu(blob_base* data, int output_channel,
		int kernel_size, int stride = 1, int pad = 0, op_name activation_op =
				CACU_RELU);

layer_block* fc_layer(blob_base* data, int output_channel, int kernel_size = 0,
		int stride = 0, int pad = 0, op_name activation_op = CACU_RELU);

layer_block* fc_layer_nodropout(blob_base* data, int output_channel,
		int kernel_size = 0, int stride = 0, int pad = 0);

layer_block* loss_layer(blob_base* data, blob_base* label, int output_channel);

layer_block* predict_layer(blob_base* data, int output_channel);



#endif