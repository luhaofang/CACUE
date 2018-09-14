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

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "../../config.h"

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"
#include "../../tensor/cuda/cuda_log.h"
#include "../../utils/log.h"

namespace cacu {

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
__global__ void _k_CACU_MAX_POOLING_CUDA(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y, int* index) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
	int start_in, in;
	int c;
	int data_row, data_col;

	int cout_length = output_w * output_h;
	int cin_length = input_w * input_h;
	int output_length = output_w * output_h * channel;
	int pad_w = abs(input_w - (output_w - 1) * stride - kernel_size);
	//int pad_h = abs(input_h - (output_h - 1) * stride - kernel_size);

	int widthx;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM) {

		data_row = ((i % cout_length) / output_w);
		data_col = ((i % cout_length) % output_w);

		c = i / cout_length;

		start_in = (data_row * input_w + data_col) * stride + c * cin_length;

		widthx = kernel_size;
		//heightx = kernel_size;
		if (data_col == output_w - 1)
			widthx -= pad_w;
		//if(data_col == output_h - 1)
		//	heightx -= pad_h;
		for (int ki = 0; ki < kernel_size && (data_row * stride + ki) < input_h;
				++ki)
			for (int kj = 0;
					kj < kernel_size && (data_col * stride + kj) < input_w;
					++kj) {
				in = start_in + ki * input_w + kj;
				if ((ki == 0 && kj == 0) || y[i] < x[in]) {
					y[i] = x[in];
					index[i] = (int) (ki * widthx + kj);
				}
			}
	}
}

extern "C" void cacu_max_pooling_cuda(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int output_w, const int output_h, const int channel, float_t *y,
		int* index) {

	_k_CACU_MAX_POOLING_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size, stride,
			input_w, input_h, output_w, output_h, channel, y, index);
	CUDA_CHECK(cudaThreadSynchronize());
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
__global__ void _k_CACU_MAX_POOLING_GRAD_CUDA(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y, const int* index) {

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

	int length = input_w * input_h * channel;

	int cin_length = input_w * input_h;

	int cout_length = output_w * output_h;

	int widthx;

	int pad_w = abs(input_w - (output_w - 1) * stride - kernel_size);

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = 0;

		startset_i = (i % cin_length) / input_w;
		startset_j = (i % cin_length) % input_w;

		c = i / cin_length;

		outset_si = startset_i / stride;
		outset_sj = startset_j / stride;

		if (outset_si >= output_h)
			outset_si = output_h - 1;
		if (outset_sj >= output_w)
			outset_sj = output_w - 1;

		count_i = 0;
		count_j = 0;

		while (outset_si - (count_i + 1) >= 0
				&& ((outset_si - (count_i + 1)) * stride) + kernel_size - 1
						>= startset_i) {
			count_i++;
		}
		while (outset_sj - (count_j + 1) >= 0
				&& ((outset_sj - (count_j + 1)) * stride) + kernel_size - 1
						>= startset_j) {
			count_j++;
		}

		for (int mi = 0; mi <= count_i; mi++)
			for (int mj = 0; mj <= count_j; mj++) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;
				if (outset_j * stride + kernel_size - 1 >= startset_j
						&& outset_i * stride + kernel_size - 1 >= startset_i) {
					widthx = kernel_size;
					if (outset_j == output_w - 1)
						widthx -= pad_w;

					offset_i = startset_i - outset_i * stride;
					offset_j = startset_j - outset_j * stride;

					if (index[(outset_i * output_w + outset_j) + c * cout_length]
							== (int) (offset_i * widthx + offset_j)) {
						y[i] += (x[(outset_i * output_w + outset_j)
								+ c * cout_length]);
					}
				}
			}

	}
}

extern "C" void cacu_max_pooling_grad_cuda(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y, const int* index) {

	_k_CACU_MAX_POOLING_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size,
			stride, input_w, input_h, output_w, output_h, channel, y, index);
	CUDA_CHECK(cudaThreadSynchronize());
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 */
__global__ void _k_CACU_AVERAGE_POOLING_CUDA(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
	int start_in, in;
	int c;
	int data_row, data_col;

	int cin_length = input_w * input_h;
	int cout_length = output_w * output_h;
	int output_length = output_w * output_h * channel;

	float_t sum;
	int count;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM) {

		data_row = ((i % cout_length) / output_w) * stride;
		data_col = ((i % cout_length) % output_w) * stride;

		c = i / cout_length;

		start_in = (data_row * input_w + data_col) + c * cin_length;

		sum = 0;
		count = 0;

		for (int ki = 0; ki < kernel_size && data_row + ki < input_h; ++ki)
			for (int kj = 0; kj < kernel_size && data_col + kj < input_w;
					++kj) {
				in = start_in + ki * input_w + kj;
				sum += x[in];
				count++;
			}
		y[i] = sum / count;
	}
}

extern "C" void cacu_average_pooling_cuda(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y) {

	_k_CACU_AVERAGE_POOLING_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size,
			stride, input_w, input_h, output_w, output_h, channel, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

/*
 *channel: channel of input data
 *kernel_size: pooling window size
 *input_dim: width of input data
 *output_dim: width of output data
 LOG_DEBUG("fuck");
 */
__global__ void _k_CACU_AVERAGE_POOLING_GRAD_CUDA(const float_t *x,
		int kernel_size, int stride, int input_w, int input_h, int output_w,
		int output_h, int channel, int pad_w, int pad_h, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;

	int length = input_w * input_h * channel;

	int cin_length = input_w * input_h;

	int cout_length = output_w * output_h;

	int pw, ph;

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = 0;

		startset_i = (i % cin_length) / input_w;
		startset_j = (i % cin_length) % input_w;

		c = i / cin_length;

		outset_si = startset_i / stride;
		outset_sj = startset_j / stride;

		if (outset_si >= output_h)
			outset_si = output_h - 1;
		if (outset_sj >= output_w)
			outset_sj = output_w - 1;

		count_i = 0;
		count_j = 0;

		while (outset_si - (count_i + 1) >= 0
				&& ((outset_si - (count_i + 1)) * stride) + kernel_size - 1
						>= startset_i) {
			count_i++;
		}
		while (outset_sj - (count_j + 1) >= 0
				&& ((outset_sj - (count_j + 1)) * stride) + kernel_size - 1
						>= startset_j) {
			count_j++;
		}

		//stride
		for (int mi = 0; mi <= count_i; mi++)
			for (int mj = 0; mj <= count_j; mj++) {
				outset_i = outset_si - mi;
				outset_j = outset_sj - mj;

				if (outset_j * stride + kernel_size - 1 >= startset_j
						&& outset_i * stride + kernel_size - 1 >= startset_i) {
					ph = kernel_size;
					pw = kernel_size;

					if (outset_i == output_h - 1)
						ph = kernel_size - pad_h;

					if (outset_j == output_w - 1)
						pw = kernel_size - pad_w;

					y[i] += (x[(outset_i * output_w + outset_j)
							+ c * cout_length] / (float_t) (ph * pw));
				}
			}
	}
}

extern "C" void cacu_average_pooling_grad_cuda(const float_t *x,
		const int kernel_size, const int stride, const int input_w,
		const int input_h, const int output_w, const int output_h,
		const int channel, float_t *y) {

	//added pad space to feature map
	int pad_w = abs(input_w - (output_w - 1) * stride - kernel_size);
	int pad_h = abs(input_h - (output_h - 1) * stride - kernel_size);

	_k_CACU_AVERAGE_POOLING_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x,
			kernel_size, stride, input_w, input_h, output_w, output_h, channel,
			pad_w, pad_h, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_IMG2COL_PAD_CUDA(const float_t *x, const int kernel_w, const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int k_row, k_col, c;

	int cin_length = input_w * input_h;
	int kernel_length = kernel_w * kernel_h;
	int block_size = channel * kernel_length;
	int output_size = output_w * output_h;
	int in_w, in_h, out_w, out_h;

	for (int j = bid; j < output_size; j += BLOCKNUM) {

		out_h = (j / output_w) * stride;
		out_w = (j % output_w) * stride;

		for (int i = tid; i < block_size; i += THREADNUM)
		{
			if (kernel_size != 1) {
				k_row = (i % kernel_length) / kernel_w;
				k_col = (i % kernel_length) % kernel_w;
			} else {
				k_row = 0;
				k_col = 0;
			}
			in_w = out_w + k_col;
			in_h = out_h + k_row;
			c = i / kernel_length;
			if (in_w >= pad_w && in_w < input_w + pad_w && in_h >= pad_h
					&& in_h < input_h + pad_h)
				y[j + i * output_size] = x[(in_h - pad_h) * input_w
						+ (in_w - pad_w) + c * cin_length];
			else
				y[j + i * output_size] = 0;
		}
	}
}

extern "C" void cacu_img2col_pad_cuda(const float_t *x, const int kernel_w, const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
	_k_CACU_IMG2COL_PAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_w, kernel_h, stride,
			input_w, input_h, channel, output_w, output_h, pad_w, pad_h, y);
	CUDA_CHECK(cudaThreadSynchronize());
}


__global__ void _k_CACU_IMG2COL_PAD_DILATED_CUDA(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, const int d_size, float_t *y) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int k_row, k_col, c;

	int cin_length = input_w * input_h;
	int kernel_length = kernel_size * kernel_size;
	int block_size = channel * kernel_length;
	int output_size = output_w * output_h;
	int in_w, in_h, out_w, out_h;
	int input_w_ = input_w + 2 * pad_w;
	int input_h_ = input_h + 2 * pad_h;

	for (int j = bid; j < output_size; j += BLOCKNUM) {

		out_h = (j / output_w) * stride;
		out_w = (j % output_w) * stride;

		for (int i = tid; i < block_size; i += THREADNUM)
		{
			if (kernel_size != 1) {
				k_row = (i % kernel_length) / kernel_size;
				k_col = (i % kernel_length) % kernel_size;
			} else {
				k_row = 0;
				k_col = 0;
			}
			in_w = out_w + k_col*d_size;
			in_h = out_h + k_row*d_size;
			in_w = (in_w >= input_w_) ? out_w + k_col : in_w;
			in_h = (in_h >= input_h_) ? out_h + k_row : in_h;
			c = i / kernel_length;
			if (in_w >= pad_w && in_w < input_w + pad_w && in_h >= pad_h
					&& in_h < input_h + pad_h)
				y[j + i * output_size] = x[(in_h - pad_h) * input_w
						+ (in_w - pad_w) + c * cin_length];
			else
				y[j + i * output_size] = 0;
		}
	}
}

extern "C" void cacu_img2col_pad_dilated_cuda(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, const int d_size, float_t *y) {
	_k_CACU_IMG2COL_PAD_DILATED_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size, stride,
			input_w, input_h, channel, output_w, output_h, pad_w, pad_h, d_size, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_COL2IMG_PAD_CUDA(const float_t *x, const int kernel_w, const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;

	int k_index, outset_index, inset_index;

	int input_w_ = input_w + 2 * pad_w;
	int input_h_ = input_h + 2 * pad_h;

	int length = input_w_ * input_h_ * channel;

	int cin_length = input_w * input_h;

	int cin_length_ = input_w_ * input_h_;

	int output_size = output_w * output_h;

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		//row
		startset_i = (i % cin_length_) / input_w_;
		//col
		startset_j = (i % cin_length_) % input_w_;

		if (startset_i >= pad_h && startset_i < input_h + pad_h
				&& startset_j >= pad_w && startset_j < input_w + pad_w) {
			//channel
			c = i / cin_length_;

			inset_index = ((startset_i - pad_h) * input_w + (startset_j - pad_w))
					+ c * cin_length;
			y[inset_index] = 0;

			outset_si = startset_i / stride;
			outset_sj = startset_j / stride;

			if (outset_si >= output_h)
				outset_si = output_h - 1;
			if (outset_sj >= output_w)
				outset_sj = output_w - 1;

			count_i = 0;
			count_j = 0;

			while (outset_si - (count_i + 1) >= 0
					&& ((outset_si - (count_i + 1)) * stride) + kernel_h - 1
							>= startset_i) {
				count_i++;
			}
			while (outset_sj - (count_j + 1) >= 0
					&& ((outset_sj - (count_j + 1)) * stride) + kernel_w - 1
							>= startset_j) {
				count_j++;
			}

			//stride
			for (int mi = 0; mi <= count_i; ++mi)
				for (int mj = 0; mj <= count_j; ++mj) {
					outset_i = outset_si - mi;
					outset_j = outset_sj - mj;

					k_index = ((startset_i - outset_i * stride) * kernel_w
							+ (startset_j - outset_j * stride))
							+ c * kernel_w * kernel_h;
					outset_index = (outset_i * output_w + outset_j);

					y[inset_index] += x[outset_index + k_index * output_size];
				}
		}
	}
}

extern "C" void cacu_col2img_pad_cuda(const float_t *x, const int kernel_w, const int kernel_h,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
	_k_CACU_COL2IMG_PAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_w, kernel_h, stride,
			input_w, input_h, channel, output_w, output_h, pad_w, pad_h, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_COL2IMG_PAD_DILATED_CUDA(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, const int d_size, float_t *y) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj, outset_i, outset_j;
	//the count for stride in feature map
	int count_i, count_j;

	int k_index, outset_index, inset_index;

	int input_w_ = input_w + 2 * pad_w;
	int input_h_ = input_h + 2 * pad_h;

	int length = input_w_ * input_h_ * channel;

	int cin_length = input_w * input_h;

	int cin_length_ = input_w_ * input_h_;

	int output_size = output_w * output_h;

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		//row
		startset_i = (i % cin_length_) / input_w_;
		//col
		startset_j = (i % cin_length_) % input_w_;

		if (startset_i >= pad_h && startset_i < input_h + pad_h
				&& startset_j >= pad_w && startset_j < input_w + pad_w) {
			//channel
			c = i / cin_length_;

			inset_index = ((startset_i - pad_h) * input_w + (startset_j - pad_w))
					+ c * cin_length;
			y[inset_index] = 0;

			outset_si = startset_i / stride;
			outset_sj = startset_j / stride;

			if (outset_si >= output_h)
				outset_si = output_h - 1;
			if (outset_sj >= output_w)
				outset_sj = output_w - 1;

			count_i = 0;
			count_j = 0;

			while (outset_si - (count_i + 1) * d_size >= 0
					&& ((outset_si - (count_i + 1)) * stride) + kernel_size - 1
							>= startset_i) {
				count_i++;
			}
			while (outset_sj - (count_j + 1) >= 0
					&& ((outset_sj - (count_j + 1)) * stride) + kernel_size - 1
							>= startset_j) {
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
					outset_index = (outset_i * output_w + outset_j);

					y[inset_index] += x[outset_index + k_index * output_size];
				}
		}
	}
}

extern "C" void cacu_col2img_pad_dilated_cuda(const float_t *x, const int kernel_size,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, const int d_size, float_t *y) {
	_k_CACU_COL2IMG_PAD_DILATED_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, kernel_size, stride,
			input_w, input_h, channel, output_w, output_h, pad_w, pad_h, d_size, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_COL2IMG_PAD_1x1_CUDA(const float_t *x,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	//the set in the input feature map
	int startset_i, startset_j;
	//the set in the output feature map
	int outset_si, outset_sj;

	int outset_index, inset_index;

	int input_w_ = input_w + 2 * pad_w;
	int input_h_ = input_h + 2 * pad_h;

	int length = input_w_ * input_h_ * channel;

	int cin_length = input_w * input_h;

	int cin_length_ = input_w_ * input_h_;

	int output_size = output_w * output_h;

	int c;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		//row
		startset_i = (i % cin_length_) / input_w_;
		//col
		startset_j = (i % cin_length_) % input_w_;

		if (startset_i >= pad_h && startset_i < input_h + pad_h
				&& startset_j >= pad_w && startset_j < input_w + pad_w) {
			//channel
			c = i / cin_length_;

			inset_index = ((startset_i - pad_h) * input_w + (startset_j - pad_w))
					+ c * cin_length;

			outset_si = startset_i / stride;
			outset_sj = startset_j / stride;

			if (startset_i % stride == 0 && startset_j % stride == 0) {

				outset_index = (outset_si * output_w + outset_sj);

				y[inset_index] += x[outset_index + c * output_size];

			}
		}
	}
}

extern "C" void cacu_col2img_pad_1x1_cuda(const float_t *x,
		const int stride, const int input_w, const int input_h,
		const int channel, const int output_w, const int output_h,
		const int pad_w, const int pad_h, float_t *y) {

	_k_CACU_COL2IMG_PAD_1x1_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, stride,
			input_w, input_h, channel, output_w, output_h, pad_w, pad_h, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _K_CACU_ROW_MAX_POOLING_CUDA(float_t *x, int input_length,
		int output_length, float_t *y) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM)
	{
		y[i] = x[i];
	}
}

extern "C" void cacu_row_max_pooling_cuda(float_t *x, int input_length,
		int output_length, float_t *y) {

	thrust::stable_sort(thrust::device, x, x + input_length,
			thrust::greater<float_t>());
	_K_CACU_ROW_MAX_POOLING_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, input_length,
			output_length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _K_CACU_ROW_MAX_POOLING_INDEX_CUDA(const float_t *x,
		int input_length, int output_length, float_t *y, int* index) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM)
	{
		for (int j = 0; j < input_length; ++j) {
			if (x[j] == y[i]) {
				index[i] = j;
				break;
			}
		}
	}
}

extern "C" void cacu_row_max_pooling_index_cuda(const float_t *x,
		int input_length, int output_length, float_t *y, int* index) {
	_K_CACU_ROW_MAX_POOLING_INDEX_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x,
			input_length, output_length, y, index);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _K_CACU_ROW_MAX_POOLING_GRAD_CUDA(const float_t *x,
		int output_length, float_t *y, const int* index) {
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < output_length; i += BLOCKNUM * THREADNUM)
	{
		y[index[i]] = x[i];
	}
}

extern "C" void cacu_row_max_pooling_grad_cuda(const float_t *x,
		int output_length, float_t *y, const int* index) {
	_K_CACU_ROW_MAX_POOLING_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x,
			output_length, y, index);
	CUDA_CHECK(cudaThreadSynchronize());
}

}

#endif
#endif
