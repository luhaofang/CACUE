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

__global__ void _k_CACU_RELU_GPU(float_t *x, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		x[i] = max((float_t) 0, x[i]);

	}
}

/**
 * for activation use relu functions in cuda
 */
extern "C" void cacu_relu_gpu(float_t *x, int length) {

	_k_CACU_RELU_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, length);

	CUDA_CHECK(cudaThreadSynchronize());

}


__global__ void _k_CACU_RELU_GRAD_GPU(float_t *x, float_t *g, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		if(x[i] <= 0)
			g[i] = 0;

	}
}

/**
 * gradient for activation use relu functions in cuda
 */
extern "C" void cacu_relu_grad_gpu(float_t *x,float_t *g, int length) {

	_k_CACU_RELU_GRAD_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, g, length);

	CUDA_CHECK(cudaThreadSynchronize());

}


__global__ void _k_CACU_LEAKY_RELU_GPU(float_t *x,float_t a, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		if(x[i] < 0)
			x[i] *= a;
	}

}

/**
 * for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_gpu(float_t *x, float_t a, int length) {

	_k_CACU_LEAKY_RELU_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, a, length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_LEAKY_RELU_GRAD_GPU(float_t *x, float_t *g, float_t a, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		if(x[i] <= 0)
			g[i] *= a;
	}

}

/**
 * gradient for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_grad_gpu(float_t *x, float_t *g, float_t a, int length) {

	_k_CACU_LEAKY_RELU_GRAD_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, g, a, length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_SOFTMAX_GPU(float_t *x, int num, int length,float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	__shared__ float_t sum[THREADNUM], max_data[THREADNUM];

	float_t *xp,*yp;

	for (int i = bid; i < num ; i += BLOCKNUM){

		xp = x + i * length;
		yp = y + i * length;

		if (tid == 0) {
			max_data[0] = xp[0];
			for (int i = 1; i < length; i++)
				max_data[0] = max(max_data[0], xp[i]);
		}

		__syncthreads();

		//bank conflict
		max_data[tid] = max_data[0];
		sum[tid] = 0;
		for (int i = tid; i < length; i += THREADNUM) {
			yp[i] = exp(xp[i] - max_data[tid]);
			sum[tid] += yp[i];
		}

		__syncthreads();

		int acc_length = THREADNUM / 2;
		while(acc_length > 0){
			if(tid < acc_length)
				sum[tid] += sum[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		//bank conflict
		sum[tid] = sum[0];

		__syncthreads();

		for (int i = tid; i < length; i += THREADNUM) {
			yp[i] /= sum[tid];
		}

	}

}

/**
 * for activation use softmax functions in cuda
 */
extern "C" void cacu_softmax_gpu(float_t *x, int num ,int length,float_t *y) {

	_k_CACU_SOFTMAX_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, num, length,y);

	CUDA_CHECK(cudaThreadSynchronize());

}

