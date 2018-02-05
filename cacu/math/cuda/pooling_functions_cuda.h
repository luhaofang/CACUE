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

#include "../../config.h"

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"

namespace cacu {

/**
 * @cacu_max_pooling_cuda
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 * index: the array of the maximum set
 */
extern "C" void cacu_max_pooling_cuda(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y,
		unsigned int* index);

/**
 * @cacu_max_pooling_cuda
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 * index: the array of the maximum set
 */
extern "C" void cacu_max_pooling_grad_cuda(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y, const unsigned int* index);

/**
 * @cacu_average_pooling_cuda
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 */
extern "C" void cacu_average_pooling_cuda(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y);

/**
 * @cacu_average_pooling_cuda
 * channel: channel of input data
 * kernel_size: pooling window size
 * input_dim: width of input data
 * output_dim: width of output data
 */
extern "C" void cacu_average_pooling_grad_cuda(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y);

extern "C" void cacu_img2col_pad_cuda(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y);

extern "C" void cacu_col2img_pad_cuda(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y);

extern "C" void cacu_col2img_pad_1x1_cuda(const float_t *x,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y);

extern "C" void cacu_row_max_pooling_cuda(float_t *x, int input_length,
		int output_length, float_t *y);

extern "C" void cacu_row_max_pooling_index_cuda(float_t *x, int input_length,
		int output_length, float_t *y, unsigned int* index);

extern "C" void cacu_row_max_pooling_grad_cuda(const float_t *x,
		int output_length, float_t *y, const unsigned int* index);

}

#endif
#endif
