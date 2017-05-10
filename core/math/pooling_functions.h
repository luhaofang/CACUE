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

#include "cuda/pooling_functions_cuda.h"
#include "cpu/pooling_functions_cpu.h"

#include "../utils/configs.h"
#include "../utils/data_defination.h"

namespace mycnn{

	/*
	 *channel: channel of input data
	 *kernel_size: pooling window size
	 *input_dim: width of input data
	 *output_dim: width of output data
	 */
	inline void cacu_max_pooling(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_max_pooling_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
#else
		cacu_max_pooling_cpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
#endif
	}


	/*
	 *channel: channel of input data
	 *kernel_size: pooling window size
	 *input_dim: width of input data
	 *output_dim: width of output data
	 */
	inline void cacu_max_pooling_grad(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_max_pooling_grad_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
#else
		cacu_max_pooling_grad_cpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
#endif
	}

	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	inline void cacu_average_pooling(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_average_pooling_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
#else
		cacu_average_pooling_cpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
#endif
	}

	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	inline void cacu_average_pooling_grad(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_average_pooling_grad_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
#else
		cacu_average_pooling_grad_cpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
#endif
	}

	/*
	*channel: channel of input data
	*input_dim: width of input data
	*pad: pad size of input data
	*/
	template<typename DTYPE>
	inline void cacu_padded_data(DTYPE *x, int channel, int input_dim, int pad, DTYPE *y)
	{
#if __PARALLELTYPE__ == __CUDA__
		cacu_padded_data_gpu(x,channel,input_dim,pad,y);
#else
		cacu_padded_data_cpu(x,channel,input_dim,pad,y);
#endif
	}

	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*stride: stride move of the kernel
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	inline void cacu_img2col(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_img2col_gpu(x,kernel_size,stride,input_dim,channel,output_dim,y);
#else
		cacu_img2col_cpu(x,kernel_size,stride,input_dim,channel,output_dim,y);
#endif
	}

	inline void cacu_img2col_pad(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, int pad, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_img2col_pad_gpu(x,kernel_size,stride,input_dim,channel,output_dim, pad,y);
#else
		cacu_img2col_pad_cpu(x,kernel_size,stride,input_dim,channel,output_dim, pad,y);
#endif
	}

	/*
	*channel: channel of input data
	*input_dim: width of input data
	*pad: pad size of input data
	*/
	template<typename DTYPE>
	inline void cacu_unpadded_data(DTYPE *x, int channel, int input_dim, int pad, DTYPE *y)
	{
#if __PARALLELTYPE__ == __CUDA__
		cacu_unpadded_data_gpu(x,channel,input_dim,pad,y);
#else
		cacu_unpadded_data_cpu(x,channel,input_dim,pad,y);
#endif
	}


	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*stride: stride move of the kernel
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	inline void cacu_col2img(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_col2img_gpu(x,kernel_size,stride,input_dim,channel,output_dim,y);
#else
		cacu_col2img_cpu(x,kernel_size,stride,input_dim,channel,output_dim,y);
#endif
	}

	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*stride: stride move of the kernel
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	inline void cacu_col2img_pad(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim,int pad, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		if(kernel_size != 1)
			cacu_col2img_pad_gpu(x,kernel_size,stride,input_dim,channel,output_dim,pad,y);
		else
			cacu_col2img_pad_1x1_gpu(x,stride,input_dim,channel,output_dim,pad,y);
#else
		cacu_col2img_pad_cpu(x,kernel_size,stride,input_dim,channel,output_dim,pad,y);
#endif
	}

	/*
	 *channel: channel of input data
	 *kernel_size: pooling window size
	 *input_dim: width of input data
	 *output_dim: width of output data
	 */
	inline void cacu_row_max_pooling(float_t *x, int input_length, int output_length, float_t *y)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_row_max_pooling_gpu(x, input_length, output_length, y);
#else
		cacu_row_max_pooling_cpu(x, input_length, output_length, y);
#endif
	}


	inline void cacu_row_max_pooling_index(float_t *x, int input_length, int output_length, float_t *y,unsigned int* index)
	{
#if __PARALLELTYPE__ == __CUDA__
		cacu_row_max_pooling_index_gpu(x, input_length, output_length, y, index);
#else
		cacu_row_max_pooling_index_cpu(x, input_length, output_length, y, index);
#endif
	}


	/*
	 *channel: channel of input data
	 *kernel_size: pooling window size
	 *input_dim: width of input data
	 *output_dim: width of output data
	 */
	inline void cacu_row_max_pooling_grad(float_t *x, int output_length, float_t *y, unsigned int* index)
	{

#if __PARALLELTYPE__ == __CUDA__
		cacu_row_max_pooling_grad_gpu(x, output_length, y, index);
#else
		cacu_row_max_pooling_grad_cpu(x, output_length, y, index);
#endif
	}
};
