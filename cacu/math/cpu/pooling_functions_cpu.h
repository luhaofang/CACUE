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

#include "../../definition.h"

namespace cacu {

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_max_pooling_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int output_dim,
		const int channel, float_t *y, unsigned int* index);

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_max_pooling_grad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int output_dim,
		const int channel, float_t *y, const unsigned int* index);

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_average_pooling_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int output_dim,
		const int channel, float_t *y);

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_average_pooling_grad_cpu(const float_t *x,
		const int kernel_size, const int stride, const int input_dim,
		const int output_dim, const int channel, float_t *y);

/*
 *channel: channel of input data
 *input_dim: width of input data
 *pad: pad size of input data
 */
template<typename DTYPE>
void cacu_padded_data_cpu(DTYPE *x, const int channel,
		const int input_dim, const int pad, DTYPE *y) {
	DTYPE *xp, *yp;
	int output_dim = input_dim + 2 * pad;
	int in_csize = input_dim * input_dim;
	int out_csize = output_dim * output_dim;
	int boundary = input_dim + pad;
	int i, j, c;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,yp,xp)
#endif
	for (c = 0; c < channel; ++c) {

		yp = y + c * out_csize;
		xp = x + c * in_csize;
		for (i = 0; i < output_dim; ++i)
			for (j = 0; j < output_dim; ++j) {
				if (i >= pad && i < boundary && j >= pad && j < boundary)
					yp[i * output_dim + j] = xp[(i - pad) * input_dim
							+ (j - pad)];
			}
	}
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *stride: stride move of the kernel
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_img2col_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int channel,
		const int output_dim, float_t *y);

void cacu_img2col_pad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int channel,
		const int output_dim, const int pad, float_t *y);

/*
 *channel: channel of input data
 *input_dim: width of input data
 *pad: pad size of input data
 */
template<typename DTYPE>
void cacu_unpadded_data_cpu(DTYPE *x, const int channel,
		const int input_dim, const int pad, DTYPE *y) {
	DTYPE *xp, *yp;
	int output_dim = input_dim - 2 * pad;
	int cin_length = input_dim * input_dim;
	int cout_length = output_dim * output_dim;

	int i, j, c;

#if __OPENMP__ == ON
#pragma omp parallel for default(shared) private(i,j,c,yp,xp)
#endif
	for (c = 0; c < channel; ++c) {
		yp = y + c * cout_length;
		xp = x + c * cin_length;
		for (i = 0; i < output_dim; ++i)
			for (j = 0; j < output_dim; ++j) {
				yp[i * output_dim + j] = xp[(i + pad) * input_dim + (j + pad)];
			}
	}
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *stride: stride move of the kernel
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_col2img_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int channel,
		const int output_dim, float_t *y);

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *stride: stride move of the kernel
 *input_dim: width of input data
 *output_dim: width of output data
 */
void cacu_col2img_pad_cpu(const float_t *x, const int kernel_size,
		const int stride, const int input_dim, const int channel,
		const int output_dim, const int pad, float_t *y);

void cacu_row_max_pooling_cpu(float_t *x, const int input_length,
		const int output_length, float_t *y);

void cacu_row_max_pooling_index_cpu(float_t *x, const int input_length,
		const int output_length, float_t *y, unsigned int* index);

void cacu_row_max_pooling_grad_cpu(const float_t *x,
		const int output_length, float_t *y, const unsigned int* index);

}
