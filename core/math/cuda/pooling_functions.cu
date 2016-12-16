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

#include "cuda_log.h"


/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
__global__ void _k_CACU_MAX_POOLING_GPU(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
	int start_in, in ;
	int c;
	int data_row, data_col;

	int cout_length = output_dim * output_dim;
	int output_length = output_dim*output_dim*channel;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM) {

		data_row = ((i%cout_length)/output_dim) * stride;
		data_col = ((i%cout_length)%output_dim) * stride;

		c = i / cout_length;

		start_in = (data_row*input_dim + data_col) + c* cout_length;

		for(int ki = 0 ; ki < kernel_size && data_row + ki < input_dim ; ++ki)
			for(int kj = 0 ; kj < kernel_size && data_col + kj < input_dim ; ++kj)
			{
				in = start_in + ki*kernel_size + kj;
				if((ki == 0 && kj ==0) || y[i] < x[in])
				{
					y[i] = x[in];
					index[i] = ki * kernel_size + kj;
				}
			}
	}
}


extern "C" void cacu_max_pooling_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index){

	_k_CACU_MAX_POOLING_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
	CUDA_CHECK(cudaThreadSynchronize());
}

/*
*channel: channel of input data
*kernel_size: pooling window size
*input_dim: width of input data
*output_dim: width of output data
*/
__global__ void _k_CACU_AVERAGE_POOLING_GPU(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
	int start_in, in ;
	int c;
	int data_row, data_col;

	int cout_length = output_dim * output_dim;
	int output_length = output_dim*output_dim*channel;

	float_t sum;
	int count;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM) {

		data_row = ((i%cout_length)/output_dim) * stride;
		data_col = ((i%cout_length)%output_dim) * stride;

		c = i / cout_length;

		start_in = (data_row*input_dim + data_col) + c* cout_length;

		sum = 0;
		count = 0;

		for(int ki = 0 ; ki < kernel_size && data_row + ki < input_dim ; ++ki)
			for(int kj = 0 ; kj < kernel_size && data_col + kj < input_dim ; ++kj)
			{
				in = start_in + ki*kernel_size + kj;
				sum += x[in];
				count ++;
			}
		y[i] = sum / count;
	}
}

extern "C" void cacu_average_pooling_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y){

	_k_CACU_AVERAGE_POOLING_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size ,stride, input_dim, output_dim ,channel, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_PADDED_DATA_GPU(float_t *x, int channel,int input_dim,int pad,float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int input_dim_ = input_dim + 2 * pad;

	int length = input_dim_ * input_dim_ * channel;

	int cout_length = input_dim_ * input_dim_;

	int cin_length = input_dim * input_dim;

	int col,row;

	int in_start , c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		row = ((i % cout_length) / input_dim_);
		col = i % input_dim_;
		c = i / cout_length;
		if (row >= pad && row <= input_dim) {
			if (col >= pad && col <= input_dim) {
				in_start = ((row - pad) * input_dim + (col - pad)) + c * cin_length;
				y[i] = x[in_start];
			} else
				y[i] = 0.0;
		} else
			y[i] = 0.0;
	}
}

extern "C" void cacu_padded_data_gpu(float_t *x,int channel, int input_dim, int pad, float_t *y){

	_k_CACU_PADDED_DATA_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, channel , input_dim, pad, y);
	CUDA_CHECK(cudaThreadSynchronize());
}


__global__ void _k_CACU_IMG2COL_GPU(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int out_start, in_start, in;

	int k_row, k_col, c;

	int cin_length = input_dim * input_dim;
	int kernel_length = kernel_size * kernel_size ;
	int block_size = channel * kernel_length;
	int output_size = output_dim * output_dim;

	for (int j = bid; j < output_size; j += BLOCKNUM) {

		out_start = j * block_size;
		in_start = ((j / output_dim) * input_dim + (j % output_dim)) * stride ;

		for (int i = tid; i < block_size; i += THREADNUM)
		{
			k_row = (i % (kernel_length)) / kernel_size;
			k_col = i % kernel_size;
			c = i / kernel_length;
			in = in_start + (k_row * input_dim + k_col) + c * cin_length;
			y[out_start + i] = x[in];

		}

		__syncthreads();

	}
}

extern "C" void cacu_img2col_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
{
	_k_CACU_IMG2COL_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size, stride, input_dim, channel, output_dim, y);
	CUDA_CHECK(cudaThreadSynchronize());
}



