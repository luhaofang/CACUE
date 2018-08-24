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

#include "../../tensor/cuda/cuda_log.h"

namespace cacu {

__global__ void _k_CACU_SAXPY_ATOMIC_CUDA(float *x, float a, float *y,
		const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		atomicAdd(y + i, a * x[i]);
	}

}

extern "C" void cacu_saxpy_atomic_cuda(float *x, float a, float *y,
		const int length) {
	_k_CACU_SAXPY_ATOMIC_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, a, y, length);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_ISAXB_CUDA(float *x, const int channel, const int width, const int height, const float a,
		int *index_, const float b, float *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
	float *xp,*yp;
	int c_length = width * height;
	int length = channel* c_length;

	if (index_[0] >= 0) {

		for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
			y[i] = x[i];
		}

		__syncthreads();

		for (int i = threadid ; i < c_length; i += BLOCKNUM * THREADNUM)
		{
			xp = x + i;
			yp = y + i;
			if (tid == 0)
				yp[index_[i] * c_length] = a * xp[index_[i] * c_length] + b;
		}
	}
	else {
		for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
			y[i] = 0;
		}
	}
}

/**
 * @cacu_isaxdb_cuda
 * y[index] = x[index]*a + b
 */
extern "C" void cacu_isaxb_cuda(float *x, const int channel, const int width, const int height, const float a,
		int *index_, const float b, float *y) {

	_k_CACU_ISAXB_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, channel, width, height, a, index_, b, y);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_ARGMAX_CUDA(float *x, const int length,
		unsigned int *index_) {

	__shared__ float shared_data[THREADNUM];

	__shared__ unsigned int index_data[THREADNUM];

	int tid = threadIdx.x;

	int max_length = THREADNUM;
	if (THREADNUM > length)
		max_length = length;

	if (tid < max_length) {
		shared_data[tid] = x[tid];
		index_data[tid] = tid;
	}

	for (unsigned int i = tid; i < length; i += THREADNUM) {
		if (x[i] > shared_data[tid]) {
			shared_data[tid] = x[i];
			index_data[tid] = i;
		}
	}

	__syncthreads();

	if (tid == 0) {

		for (int i = 1; i < max_length; ++i) {
			if (shared_data[0] < shared_data[i]) {
				shared_data[0] = shared_data[i];
				index_data[0] = index_data[i];
			}
		}
		index_[0] = index_data[0];
	}
}

extern "C" void cacu_argmax_cuda(float *x, const int length,
		unsigned int *index_) {
	_k_ARGMAX_CUDA<<<1, THREADNUM, 0>>>(x, length, index_);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_TRANSPOSE_CUDA(float *mtx, const int m, const int n,
		const int length) {


}


extern "C" void cacu_transpose_cuda(float *mtx, const int m, const int n, const int length) {

}

__global__ void _k_CACU_CLIP_VEC_CUDA(float *data, const float threshold,
		const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		data[i] = data[i] * (abs(data[i]) >= threshold);
	}
}

extern "C" void cacu_clip_vec_cuda(float *data, const float threshold,
		const int length) {
	_k_CACU_CLIP_VEC_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(data, threshold, length);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_ABS_CUDA(float *x, const int length, float *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = abs(x[i]);
	}
}

extern "C" void cacu_abs_cuda(float *x, const int length, float *y)
{
	_k_CACU_ABS_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_ABS_GRAD_CUDA(float *x, float *diff, const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		diff[i] = (x[i] > 0) ? 1 : -1;
	}
}

extern "C" void cacu_abs_grad_cuda(float *x, float *diff, const int length)
{
	_k_CACU_ABS_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, diff, length);

	CUDA_CHECK(cudaThreadSynchronize());
}

}

#endif
#endif
