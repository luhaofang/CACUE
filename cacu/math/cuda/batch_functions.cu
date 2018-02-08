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

#include "../../config.h"

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"
#include "../math_definition.h"
#include "../../tensor/cuda/cuda_log.h"

namespace cacu {

__global__ void _k_CACU_SUMBYSIZE_BYWIDTH_CUDA(const float_t *x, int heigth,
		int width, const float_t alpha, float_t *y, const float_t beta) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t shared_data[];

	for (int i = bid; i < heigth; i += BLOCKNUM) {
		shared_data[tid] = 0;
		for (int j = tid; j < width; j += THREADNUM) {
			shared_data[tid] += x[i * width + j];
		}
		__syncthreads();

		int acc_length = THREADNUM / 2;
		while (acc_length > 0) {
			if (tid < acc_length)
				shared_data[tid] += shared_data[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		if (tid == 0)
			y[i] = alpha * shared_data[0] + beta * y[i];
	}
}

__global__ void _k_CACU_SUMBYSIZE_BYHEIGHT_CUDA(const float_t *x, int height,
		int width, const float_t alpha, float_t *y, const float_t beta) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t shared_data[];

	for (int i = bid; i < width; i += BLOCKNUM) {
		shared_data[tid] = 0;
		for (int j = tid; j < height; j += THREADNUM)
		{
			shared_data[tid] += x[j * width + i];
		}
		__syncthreads();

		int acc_length = THREADNUM / 2;
		while (acc_length > 0) {

			if (tid < acc_length)
				shared_data[tid] += shared_data[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		if (tid == 0)
			y[i] = alpha * shared_data[0] + beta * y[i];
	}
}

/**
 * (need FULLY tested!)
 * @cacu_sumbysize_cuda
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 */
extern "C" void cacu_sumbysize_cuda(SUM SUMTYPE, const float_t *x, int length,
		const float_t alpha, float_t *y, const float_t beta, int width) {

	int height = length / width;

	if (BYWIDTH == SUMTYPE)
		_k_CACU_SUMBYSIZE_BYWIDTH_CUDA<<<BLOCKNUM, THREADNUM,
		THREADNUM * sizeof(float_t)>>>(x, height, width, alpha, y, beta);
	else if (BYHEIGHT == SUMTYPE)
		_k_CACU_SUMBYSIZE_BYHEIGHT_CUDA<<<BLOCKNUM, THREADNUM,
		THREADNUM * sizeof(float_t)>>>(x, height, width, alpha, y, beta);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_CXSIZE_CUDA(const float_t *x, int length,
		const float_t *a, int size, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int block_size = length / size;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i] * a[i / block_size];
	}
}

/**
 * @cacu_cxsize_cuda
 * math y[i] = a[j]*x[i] :
 * x: length dim array list
 * a: size dim array list
 * a[j] is the corresponding scalar, j = i / (length / size).
 */
extern "C" void cacu_cxsize_cuda(const float_t *x, int length, const float_t *a,
		int size, float_t *y) {
	_k_CACU_CXSIZE_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, a, size, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_CDXSIZE_CUDA(const float_t *x, int length,
		const float_t *a, int size, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int block_size = length / size;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i] / a[i / block_size];
	}
}

/**
 * @cacu_cdxsize_cuda
 * math y[i] = x[i] / a[j] :
 * x: length dim array list
 * a: size dim array list
 * a[j] is the corresponding denominator, j = i / (length / size).
 */
extern "C" void cacu_cdxsize_cuda(const float_t *x, int length,
		const float_t *a, int size, float_t *y) {
	_k_CACU_CDXSIZE_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, a, size, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SDXSIZE_CUDA(float_t *x, int length,
		const float_t a, const float_t b, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = (x[i] + a) / b;
	}
}

/**
 * @cacu_sdxsize
 * math y[i] = (x[i] + a) / b:
 * x is a length dim array list, a is the corresponding denominator.
 */
extern "C" void cacu_sdxsize_cuda(float_t *x, const int length,
		const float_t a, const float_t b, float_t *y) {
	_k_CACU_SDXSIZE_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, a, b, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SSXPY_CUDA(const float_t *x, const float_t a, int size,
		const float_t *y, const float_t b, int length, float_t *z) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int block_size = length / size;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		z[i] = b * y[i] + a * x[i / block_size];
	}
}

/**
 * @cacu_ssxpy_cuda
 * math z[i] = a * x[j] + b * y[i] :
 * y: length dim array list
 * x: size dim array list
 * x[j] is the corresponding scalar, j = i / (length / size).
 * a & b are corresponding scalars for x, y
 */
extern "C" void cacu_ssxpy_cuda(const float_t *x, const float_t a, int size,
		const float_t *y, const float_t b, int length, float_t *z) {
	_k_CACU_SSXPY_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, a, size, y, b, length, z);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SQR_CUDA(const float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i] * x[i];
	}
}

/**
 * @cacu_sqr_cuda
 * math y[i] = x[i]^2 :
 */
extern "C" void cacu_sqr_cuda(const float_t *x, int length, float_t *y) {
	_k_CACU_SQR_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_ROOT_CUDA(const float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = sqrt(x[i]);
	}
}

/**
 * @cacu_root_cuda
 * math y[i] = sqrt(x[i]) :
 */
extern "C" void cacu_root_cuda(const float_t *x, int length, float_t *y) {
	_k_CACU_ROOT_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_STDBYCHANNEL_CUDA(const float_t *varience, int length,
		float_t *std, const float_t epsilon) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		std[i] = sqrt(varience[i] + epsilon);
	}
}

/**
 * @cacu_stdbychannel_cuda
 * math std[i] = sqrt(varience[i] + epsilon) :
 */
extern "C" void cacu_stdbychannel_cuda(const float_t *varience, int length,
		float_t *std, const float_t epsilon) {
	_k_CACU_STDBYCHANNEL_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(varience, length, std,
			epsilon);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_BN_ROU_GRAD_CUDA(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, int num, int length,
		int channel, float_t *d_rou) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t shared_data[];

	int data_row, data_col;

	int cin_length = length / channel;

	int set;

	for (int i = bid; i < channel; i += BLOCKNUM)
	{
		shared_data[tid] = 0;
		for (int j = tid; j < cin_length * num; j += THREADNUM)
		{
			data_row = j / cin_length;
			data_col = j % cin_length;
			set = data_row * length + data_col + i * cin_length;
			shared_data[tid] += ((x[set] - mean[i]) * d_x[set]
					* (float_t(-0.5) / (std[i] * std[i] * std[i])));
		}

		__syncthreads();

		int acc_length = THREADNUM / 2;
		while (acc_length > 0) {
			if (tid < acc_length)
				shared_data[tid] += shared_data[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		if (tid == 0)
			d_rou[i] += shared_data[0];
	}
}

/**
 * @cacu_bn_rou_grad_cuda
 * calculate the gradient of bn layer's rou
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * length: size of a feature map
 * d_rou: gradient of batch's variance
 */
extern "C" void cacu_bn_rou_grad_cuda(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, int num, int length,
		int channel, float_t *d_rou) {

	_k_CACU_BN_ROU_GRAD_CUDA<<<BLOCKNUM, THREADNUM,
	THREADNUM * sizeof(float_t)>>>(x, d_x, mean, std, num, length, channel,
			d_rou);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_BN_MU_GRAD_CUDA(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, const float_t *d_rou, int num,
		int length, int channel, float_t *d_mean) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t shared_data[];

	int data_row, data_col;

	int cin_length = length / channel;

	int set;

	int m = cin_length * num;

	for (int i = bid; i < channel; i += BLOCKNUM)
	{
		shared_data[tid] = 0;
		for (int j = tid; j < cin_length * num; j += THREADNUM)
		{
			data_row = j / cin_length;
			data_col = j % cin_length;
			set = data_row * length + data_col + i * cin_length;
			shared_data[tid] += ((d_x[set] / (-std[i]))
					+ ((d_rou[i] / m) * (float_t(-2) * (x[set] - mean[i]))));
		}

		__syncthreads();

		int acc_length = THREADNUM / 2;
		while (acc_length > 0) {
			if (tid < acc_length)
				shared_data[tid] += shared_data[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		if (tid == 0)
			d_mean[i] += shared_data[0];
	}
}

/**
 * @cacu_bn_mu_grad
 * calculate the gradient of bn layer's mu
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * d_rou: gradient of batch's variance
 * length: size of a feature map
 * d_mean: gradient of batch's mean
 */
extern "C" void cacu_bn_mu_grad_cuda(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, const float_t *d_rou, int num,
		int length, int channel, float_t *d_mean) {
	_k_CACU_BN_MU_GRAD_CUDA<<<BLOCKNUM, THREADNUM,
	THREADNUM * sizeof(float_t)>>>(x, d_x, mean, std, d_rou, num, length,
			channel, d_mean);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_BN_DX_GRAD_CUDA(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, const float_t *d_rou,
		const float_t *d_mean, int num, int length, int channel, float_t *dx) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int c;

	int cin_length = length / channel;

	int m = cin_length * num;

	for (int i = threadid; i < num * length; i += BLOCKNUM * THREADNUM) {

		c = (i % length) / cin_length;
		dx[i] = ((d_x[i] / std[c])
				+ d_rou[c] * (float_t(2) * (x[i] - mean[c]) / m)
				+ (d_mean[c] / m));
	}
}

/**
 * @cacu_bn_dx_grad_cuda
 * calculate the gradient of bn layer's dx
 * x: input feature
 * d_x: gradient of ^x
 * mean: mean of batch
 * std: standard deviation of batch
 * d_rou: gradient of batch's variance
 * d_mean: gradient of batch's mean
 * length: size of a feature map
 * dx: gradient of x
 */
extern "C" void cacu_bn_dx_grad_cuda(const float_t *x, const float_t *d_x,
		const float_t *mean, const float_t *std, const float_t *d_rou,
		const float_t *d_mean, int num, int length, int channel, float_t *dx) {
	_k_CACU_BN_DX_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, d_x, mean, std,
			d_rou, d_mean, num, length, channel, dx);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_BN_GAMMA_GRAD_CUDA(const float_t *_x,
		const float_t *d_y, int num, int length, int channel,
		float_t *d_gamma) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	extern __shared__ float_t shared_data[];

	int data_row, data_col;

	int cin_length = length / channel;

	int set;

	for (int i = bid; i < channel; i += BLOCKNUM)
	{
		shared_data[tid] = 0;

		for (int j = tid; j < cin_length * num; j += THREADNUM)
		{
			data_row = j / cin_length;
			data_col = j % cin_length;
			set = data_row * length + data_col + i * cin_length;
			shared_data[tid] += (_x[set] * d_y[set]);
		}

		__syncthreads();

		int acc_length = THREADNUM / 2;
		while (acc_length > 0) {
			if (tid < acc_length)
				shared_data[tid] += shared_data[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		if (tid == 0)
			d_gamma[i] += shared_data[0];
	}
}

/**
 * @cacu_bn_gamma_grad_cuda
 * calculate the gradient of bn layer's scale
 * _x: is ^x
 * d_y: gradient propagate form top layer
 * length: size of a feature map
 * d_gamma: gradient of gamma
 */
extern "C" void cacu_bn_gamma_grad_cuda(const float_t *_x, const float_t *d_y,
		int num, int length, int channel, float_t *d_gamma) {
	_k_CACU_BN_GAMMA_GRAD_CUDA<<<BLOCKNUM, THREADNUM,
	THREADNUM * sizeof(float_t)>>>(_x, d_y, num, length, channel, d_gamma);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SSX_CUDA(const float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] *= x[i];
	}
}

/**
 * @cacu_ssx_cuda
 * math y[i] *= x[i] :
 * scale by element wise.
 */
extern "C" void cacu_ssx_cuda(const float_t *x, int length, float_t *y) {
	_k_CACU_SSX_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_GROUP_ALLOC_CUDA(int num, int channel,
		int channel_length, int group, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int length = channel * channel_length;

	int start_set = (channel / group) * channel_length;

	int copy_length = length - start_set;

	float_t *yp, *xp;

	for (int n = bid; n < num; n += BLOCKNUM) {

		yp = y + n * length + start_set;
		xp = y + n * length;

		for (int i = tid; i < copy_length; i += THREADNUM) {
			yp[i] = xp[i % start_set];
		}
	}
}

/**
 * @cacu_group_alloc
 * alloc data by group
 */
extern "C" void cacu_group_alloc_cuda(int num, int channel, int channel_length,
		int group, float_t *y) {

	_k_CACU_GROUP_ALLOC_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(num, channel,
			channel_length, group, y);
	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_GROUP_COMBINE_CUDA(int num, int channel,
		int channel_length, int group, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int length = channel * channel_length;

	int start_set = (channel / group) * channel_length;

	float_t *yp, *xp;

	for (int n = bid; n < num; n += BLOCKNUM) {

		yp = y + n * length + start_set;
		xp = y + n * length;

		for (int i = tid; i < start_set; i += THREADNUM) {
			for (int g = 0; g < group - 1; ++g)
				xp[i] += yp[i + g * start_set];
		}
	}
}

/**
 * @cacu_group_combine
 * combine data by group
 */
extern "C" void cacu_group_combine_cuda(int num, int channel,
		int channel_length, int group, float_t *y) {
	_k_CACU_GROUP_COMBINE_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(num, channel,
			channel_length, group, y);
	CUDA_CHECK(cudaThreadSynchronize());

}

}
#endif
#endif
