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


namespace mycnn{

	/*
	 *channel: channel of input data
	 *kernel_size: pooling window size
	 *input_dim: width of input data
	 *output_dim: width of output data
	 */
	void cacu_max_pooling(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_max_pooling_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
#else
		int block_size = output_dim*output_dim;
		float_t *xp, *yp, xd;
		unsigned int *ip;
		int in_start, out_start;
		for (int c = 0; c < channel; ++c)
		{
			xp = x + c*input_dim*input_dim;
			yp = y + c*block_size;
			ip = index + c*block_size;
			for (int i = 0; i < output_dim; ++i)
				for (int j = 0; j < output_dim; ++j)
				{
					out_start = (i * output_dim + j);
					in_start = (i * input_dim + j)*stride;
					for (int ki = 0; ki < kernel_size && (ki + i*stride) < input_dim; ki++)
						for (int kj = 0; kj < kernel_size && (kj + j*stride) < input_dim; kj++)
						{
							xd = xp[in_start + ki * input_dim + kj];
							if (yp[out_start] < xd || (ki == 0 && kj == 0))
							{
								yp[out_start] = xd;
								ip[out_start] = in_start + ki * input_dim + kj;
							}
						}
				}
		}
#endif
	}


	/*
	 *channel: channel of input data
	 *kernel_size: pooling window size
	 *input_dim: width of input data
	 *output_dim: width of output data
	 */
	void cacu_max_pooling_grad(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index)
	{

#if __PARALLELTYPE__ == __GPU__
		//cacu_max_pooling_grad_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
#else
		int block_size = output_dim*output_dim;
		float_t *xp, *yp, xd;
		unsigned int *ip;
		int in_start, out_start;
		for (int c = 0; c < channel; ++c)
		{
			xp = x + c*input_dim*input_dim;
			yp = y + c*block_size;
			ip = index + c*block_size;
			for (int i = 0; i < output_dim; ++i)
				for (int j = 0; j < output_dim; ++j)
				{
					out_start = (i * output_dim + j);
					in_start = (i * input_dim + j)*stride;
					for (int ki = 0; ki < kernel_size && (ki + i*stride) < input_dim; ki++)
						for (int kj = 0; kj < kernel_size && (kj + j*stride) < input_dim; kj++)
						{
							xd = xp[in_start + ki * input_dim + kj];
							if (yp[out_start] < xd || (ki == 0 && kj == 0))
							{
								yp[out_start] = xd;
								ip[out_start] = in_start + ki * input_dim + kj;
							}
						}
				}
		}
#endif
	}

	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	void cacu_average_pooling(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_average_pooling_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
#else
		int block_size = output_dim*output_dim;
		float_t *xp, *yp;
		int in_start, out_start;
		int count;
		for (int c = 0; c < channel; ++c)
		{
			xp = x + c*input_dim*input_dim;
			yp = y + c*block_size;
			for (int i = 0; i < output_dim; ++i)
				for (int j = 0; j < output_dim; ++j)
				{
					out_start = (i * output_dim + j);
					in_start = (i * input_dim + j)*stride;
					count = 0;
					for (int ki = 0; ki < kernel_size && (ki + i*stride) < input_dim; ki++)
						for (int kj = 0; kj < kernel_size && (kj + j*stride) < input_dim; kj++)
						{
							yp[out_start] += xp[in_start + ki * input_dim + kj];
							count++;
						}
					yp[out_start] /= count;
				}
		}
#endif
	}

	/*
	*channel: channel of input data
	*kernel_size: pooling window size
	*input_dim: width of input data
	*output_dim: width of output data
	*/
	void cacu_average_pooling_grad(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y)
	{

#if __PARALLELTYPE__ == __GPU__
		//cacu_average_pooling_grad_gpu(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
#else
		int block_size = output_dim*output_dim;
		float_t *xp, *yp;
		int in_start, out_start;
		int count;
		for (int c = 0; c < channel; ++c)
		{
			xp = x + c*input_dim*input_dim;
			yp = y + c*block_size;
			for (int i = 0; i < output_dim; ++i)
				for (int j = 0; j < output_dim; ++j)
				{
					out_start = (i * output_dim + j);
					in_start = (i * input_dim + j)*stride;
					count = 0;
					for (int ki = 0; ki < kernel_size && (ki + i*stride) < input_dim; ki++)
						for (int kj = 0; kj < kernel_size && (kj + j*stride) < input_dim; kj++)
						{
							yp[out_start] += xp[in_start + ki * input_dim + kj];
							count++;
						}
					yp[out_start] /= count;
				}
		}
#endif
	}

	template<typename DTYPE>
	void cacu_padded_data(DTYPE *x,int channel, int input_dim, int pad, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_padded_data_gpu(x,channel,input_dim,pad,y);
#else
		DTYPE *xp, *yp;
		int output_dim = input_dim + 2 * pad;
		int in_csize = input_dim*input_dim;
		int out_csize = output_dim*output_dim;
		int boundary = input_dim + pad;
		for (int c = 0; c < channel; ++c){

			yp = y + c*out_csize;
			xp = x + c*in_csize;
			for (int i = 0; i < output_dim; ++i)
				for (int j = 0; j < output_dim; ++j)
				{
					if (i >= pad && i < boundary && j >= pad && j<boundary)
						yp[i * output_dim + j] = xp[(i - pad)*input_dim + (j - pad)];
				}
		}
#endif
	}

	void cacu_img2col(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_img2col_gpu(x,kernel_size,stride,input_dim,channel,output_dim,y);
#else
		int block_size = input_dim*input_dim;
		int kernel_length = kernel_size*kernel_size;
		float_t *xp, *yp;
		int in_start, out_start;

		for (int i = 0; i < output_dim; ++i)
			for (int j = 0; j < output_dim; ++j)
			{
				out_start = (i * output_dim + j)*kernel_length*channel;
				in_start = (i * input_dim + j)*stride;
				
				for (int c = 0; c < channel; ++c)
				{
					yp = y + out_start + c*kernel_length;
					xp = x + in_start + c*block_size;

					for (int ki = 0; ki < kernel_size; ++ki)
						for (int kj = 0; kj < kernel_size; ++kj)
						{
							yp[ki*kernel_size + kj] = xp[ki * input_dim + kj];
						}
				}
			}
#endif
	}

	template<typename DTYPE>
	void cacu_unpadded_data(DTYPE *x,int channel, int input_dim, int pad, DTYPE *y)
	{
#if __PARALLELTYPE__ == __GPU__
		cacu_unpadded_data_gpu(x,channel,input_dim,pad,y);
#else
		DTYPE *xp, *yp;
		int output_dim = input_dim - 2 * pad;
		int in_csize = input_dim*input_dim;
		int out_csize = output_dim*output_dim;
		for (int c = 0; c < channel; ++c){

			yp = y + c*out_csize;
			xp = x + c*in_csize;
			for (int i = 0; i < output_dim; ++i)
				for (int j = 0; j < output_dim; ++j)
				{
					yp[i * output_dim + j] = xp[(i + pad)*input_dim + (j + pad)];
				}
		}
#endif
	}

	void cacu_col2img(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
	{

#if __PARALLELTYPE__ == __GPU__
		cacu_col2img_gpu(x,kernel_size,stride,input_dim,channel,output_dim,y);
#else

		float_t *sdp, *snp;
		int sd_out, sn_out;

		int block_size = kernel_size * kernel_size * channel;
		int k_size = kernel_size * kernel_size;
		int length = output_dim * output_dim * channel * kernel_size * kernel_size;
		int out_dim = output_dim * output_dim;
		int border = input_dim - output_dim;
		int in_dim = input_dim * input_dim;

		for (int num = 0; num < data.size(); num++) {

			sdp = x;
			snp = y;

			//for output_dim's location
			for (int index = 0; index < out_dim; index++) {
				sd_out = index * block_size;
				sn_out = ((index / output_dim) * input_dim + (index % output_dim))
						* stride;
				for (int ki = 0; ki < kernel_size; ki++)
					for (int kj = 0; kj < kernel_size; kj++) {
						for (int c = 0; c < channel; c++) {
							*(snp + sn_out + (ki * input_dim + kj) + c * in_dim) +=
									*(sdp + sd_out + c * k_size + ki * kernel_size
											+ kj);
						}
					}
			}
		}
#endif
	}
};
