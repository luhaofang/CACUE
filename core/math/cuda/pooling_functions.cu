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
	int cin_length = input_dim * input_dim;
	int output_length = output_dim*output_dim * channel;
	int pad = abs(input_dim - (output_dim - 1) * stride - kernel_size);

	int widthx;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM) {

		data_row = ((i%cout_length)/output_dim);
		data_col = ((i%cout_length)%output_dim);

		c = i / cout_length;

		start_in = (data_row * input_dim + data_col) * stride + c * cin_length;

		if(data_col == output_dim - 1)
			widthx = kernel_size - pad;

		for(int ki = 0 ; ki < kernel_size && (data_row*stride + ki) < input_dim ; ++ki)
			for(int kj = 0 ; kj < kernel_size && (data_col*stride + kj) < input_dim ; ++kj)
			{
				in = start_in + ki * input_dim + kj;
				if((ki == 0 && kj == 0) || y[i] < x[in])
				{
					y[i] = x[in];
					index[i] = (unsigned int)(ki * widthx + kj);
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
__global__ void _k_CACU_MAX_POOLING_GRAD_GPU(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;
	//the index for the data in kernel
	int offset_i, offset_j;

	int length = input_dim * input_dim * channel;

	int cin_length = input_dim * input_dim;

	int cout_length = output_dim * output_dim;

	int widthx;

	int pad = abs(input_dim - (output_dim - 1) * stride - kernel_size);

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		startset_i = (i % cin_length) / input_dim;
		startset_j = (i % cin_length) % input_dim;
		c = i / cin_length;
		outset_si = startset_i / stride;
		outset_sj = startset_j / stride;

		if (outset_si >= output_dim)
			outset_si = output_dim - 1;
		if (outset_sj >= output_dim)
			outset_sj = output_dim - 1;

		count_i = 0;
		count_j = 0;

		while (outset_si - (count_i + 1) >= 0
				&& ((outset_si - (count_i + 1)) * stride) + kernel_size
						> startset_i) {
			count_i++;
		}
		while (outset_sj - (count_j + 1) >= 0
				&& ((outset_sj - (count_j + 1)) * stride) + kernel_size
						> startset_j) {
			count_j++;
		}

		for (int mi = 0; mi <= count_i; mi++)
			for (int mj = 0; mj <= count_j; mj++) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;

				if(outset_j == output_dim - 1)
					widthx = kernel_size - pad;

				offset_i = startset_i - outset_i * stride;
				offset_j = startset_j - outset_j * stride;
				if (index[(outset_i * output_dim + outset_j) + c * cout_length]
						== (float_t) (offset_i * widthx + offset_j)) {
					y[i] +=	x[(outset_i * output_dim + outset_j) + c * cout_length];
				}
			}

	}
}


extern "C" void cacu_max_pooling_grad_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y, unsigned int* index){

	_k_CACU_MAX_POOLING_GRAD_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size ,stride, input_dim, output_dim ,channel, y, index);
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

	int cin_length = input_dim*input_dim;
	int cout_length = output_dim * output_dim;
	int output_length = output_dim * output_dim * channel;

	float_t sum;
	int count;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM) {

		data_row = ((i%cout_length)/output_dim) * stride;
		data_col = ((i%cout_length)%output_dim) * stride;

		c = i / cout_length;

		start_in = (data_row*input_dim + data_col) + c* cin_length;

		sum = 0;
		count = 0;

		for(int ki = 0 ; ki < kernel_size && data_row + ki < input_dim ; ++ki)
			for(int kj = 0 ; kj < kernel_size && data_col + kj < input_dim ; ++kj)
			{
				in = start_in + ki*input_dim + kj;
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

/*
*channel: channel of input data
*kernel_size: pooling window size
*input_dim: width of input data
*output_dim: width of output data
*/
__global__ void _k_CACU_AVERAGE_POOLING_GRAD_GPU(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel,int pad, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;

	int length = input_dim * input_dim * channel;

	int cin_length = input_dim * input_dim;

	int cout_length = output_dim * output_dim;

	int pw, ph;

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = 0;

		startset_i = (i % cin_length) / input_dim;
		startset_j = (i % cin_length) % input_dim;

		c = i / cin_length;

		outset_si = startset_i / stride;
		outset_sj = startset_j / stride;

		if (outset_si >= output_dim)
			outset_si = output_dim - 1;
		if (outset_sj >= output_dim)
			outset_sj = output_dim - 1;

		count_i = 0;
		count_j = 0;

		while (outset_si - (count_i + 1) >= 0
				&& ((outset_si - (count_i + 1)) * stride) + kernel_size
						> startset_i) {
			count_i++;
		}
		while (outset_sj - (count_j + 1) >= 0
				&& ((outset_sj - (count_j + 1)) * stride) + kernel_size
						> startset_j) {
			count_j++;
		}

		//stride
		for (int mi = 0; mi <= count_i; mi++)
			for (int mj = 0; mj <= count_j; mj++) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;

				pw = kernel_size;
				ph = kernel_size;

				if (outset_i == output_dim - 1)
					ph = kernel_size - pad;

				if (outset_j == output_dim - 1)
					pw = kernel_size - pad;

				y[i] +=	(x[(outset_i * output_dim + outset_j) + c*cout_length] / (float_t) (ph * pw));
			}
	}
}

extern "C" void cacu_average_pooling_grad_gpu(float_t *x, int kernel_size, int stride, int input_dim, int output_dim, int channel, float_t *y){

	//added pad space to feature map
	int pad = abs(input_dim - (output_dim - 1) * stride - kernel_size);

	_k_CACU_AVERAGE_POOLING_GRAD_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size ,stride, input_dim, output_dim ,channel, pad, y);
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

	int out_start, in_start;

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
			k_col = (i % (kernel_length)) % kernel_size;
			c = i / kernel_length;
			y[out_start + i] = x[in_start + (k_row * input_dim + k_col) + c * cin_length];

		}

		//__syncthreads();

	}
}

extern "C" void cacu_img2col_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
{
	_k_CACU_IMG2COL_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size, stride, input_dim, channel, output_dim, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_UNPADDED_DATA_GPU(float_t *x,int channel,int input_dim,int pad,float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int input_dim_ = input_dim - 2 * pad;

	int length = input_dim_ * input_dim_ * channel;

	int cout_length = input_dim_ * input_dim_;

	int cin_length = input_dim * input_dim;

	int col,row;

	int in_start , c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		row = ((i % cout_length) / input_dim_);
		col = i % input_dim_;
		c = i / cout_length;

		in_start = ((row + pad) * input_dim + (col + pad)) + c * cin_length;
		y[i] = x[in_start];
	}
}

extern "C" void cacu_unpadded_data_gpu(float_t *x,int channel, int input_dim, int pad, float_t *y){

	_k_CACU_UNPADDED_DATA_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, channel, input_dim, pad, y);
	CUDA_CHECK(cudaThreadSynchronize());
}


__global__ void _k_CACU_COL2IMG_GPU(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;

	int k_index, outset_index;

	int block_size = kernel_size * kernel_size * channel;

	int length = input_dim * input_dim * channel;

	int cin_length = input_dim * input_dim;

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = 0;
		//row
		startset_i = (i % cin_length) / input_dim;
		//col
		startset_j = (i % cin_length) % input_dim;
		//channel
		c = i / cin_length;

		outset_si = startset_i / stride;
		outset_sj = startset_j / stride;

		if (outset_si >= output_dim)
			outset_si = output_dim - 1;
		if (outset_sj >= output_dim)
			outset_sj = output_dim - 1;

		count_i = 0;
		count_j = 0;

		while (outset_si - (count_i + 1) >= 0
				&& ((outset_si - (count_i + 1)) * stride) + kernel_size
						> startset_i) {
			count_i++;
		}
		while (outset_sj - (count_j + 1) >= 0
				&& ((outset_sj - (count_j + 1)) * stride) + kernel_size
						> startset_j) {
			count_j++;
		}

		//stride
		for (int mi = 0; mi <= count_i; ++mi)
			for (int mj = 0; mj <= count_j; ++mj) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;

				k_index = ((startset_i - outset_i * stride) * kernel_size
						+ (startset_j - outset_j * stride))
						+ c * kernel_size * kernel_size;
				outset_index = (outset_i * output_dim + outset_j) * block_size;

				y[i] += x[outset_index + k_index];

			}
	}
}

extern "C" void cacu_col2img_gpu(float_t *x, int kernel_size, int stride, int input_dim, int channel, int output_dim, float_t *y)
{

	_k_CACU_COL2IMG_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size, stride, input_dim, channel, output_dim, y);
	CUDA_CHECK(cudaThreadSynchronize());
}



