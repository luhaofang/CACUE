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

#ifndef POOLING_FUNCTIONS_H_
#define POOLING_FUNCTIONS_H_

#include "../definition.h"

#include "cuda/pooling_functions_cuda.h"
#include "cpu/pooling_functions_cpu.h"

namespace cacu {

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
inline void cacu_max_pooling(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y,
		int* index) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_max_pooling_cuda(x, kernel_size, stride, input_w, input_h, output_w,
			output_h, channel, y, index);
#endif
#else
	cacu_max_pooling_cpu(x, kernel_size, stride, input_w, input_h, output_w,
			output_h, channel, y, index);
#endif
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
inline void cacu_max_pooling_grad(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y,
		const int* index) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_max_pooling_grad_cuda(x, kernel_size, stride, input_w, input_h,
			output_w, output_h, channel, y, index);
#endif
#else
	cacu_max_pooling_grad_cpu(x, kernel_size, stride, input_w, input_h,
			output_w, output_h, channel, y, index);
#endif
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
inline void cacu_average_pooling(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_average_pooling_cuda(x, kernel_size, stride, input_w, input_h,
			output_w, output_h, channel, y);
#endif
#else
	cacu_average_pooling_cpu(x, kernel_size, stride, input_w, input_h, output_w,
			output_h, channel, y);
#endif
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
inline void cacu_average_pooling_grad(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_average_pooling_grad_cuda(x, kernel_size, stride, input_w, input_h,
			output_w, output_h, channel, y);
#endif
#else
	cacu_average_pooling_grad_cpu(x, kernel_size, stride, input_w, input_h,
			output_w, output_h, channel, y);
#endif
}


inline void cacu_img2col_pad(const float_t *x, const int kernel_w, const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_img2col_pad_cuda(x, kernel_w,kernel_h, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
#endif
#else
	cacu_img2col_pad_cpu(x, kernel_w, kernel_h, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
#endif
}

inline void cacu_img2col_pad_dilated(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, const int d_size, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	cacu_img2col_pad_dilated_cuda(x, kernel_size, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, d_size, y);
#endif
#else
	cacu_img2col_pad_dilated_cpu(x, kernel_size, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, d_size, y);
#endif
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *stride: stride move of the kernel
 *input_dim: width of input data
 *output_dim: width of output data
 */
inline void cacu_col2img_pad(const float_t *x, const int kernel_w,const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	if (kernel_w != 1 && kernel_h != 1)
	cacu_col2img_pad_cuda(x, kernel_w,kernel_h, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
	else
	cacu_col2img_pad_1x1_cuda(x, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
#endif
#else
	//cout << kernel_w << "," << kernel_h << "," << stride << "," << input_w << "," << input_h << "," << output_w << "," << output_h << endl;
	cacu_col2img_pad_cpu(x, kernel_w, kernel_h, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
#endif
}

inline void cacu_col2img_pad_dilated(const float_t *x, const int kernel_w,const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, const int d_size, float_t *y) {
#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__
	if (kernel_w != 1 && kernel_h != 1)
	cacu_col2img_pad_cuda(x, kernel_w, kernel_h, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
	else
	cacu_col2img_pad_1x1_cuda(x, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, y);
#endif
#else
	cacu_col2img_pad_dilated_cpu(x, kernel_w, stride, input_w, input_h, channel,
			output_w, output_h, pad_w, pad_h, d_size, y);
#endif
}

}

#endif
