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
#include "../../tensor/cuda/cuda_log.h"

namespace cacu {

__global__ void _k_CACU_RELU_CUDA(float_t *x, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		x[i] = max(x[i], float_t(0));
	}
}

/**
 * for activation use relu functions in cuda
 */
extern "C" void cacu_relu_cuda(float_t *x, int length) {

	_k_CACU_RELU_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_RELU_GRAD_CUDA(float_t *x, float_t *g, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		g[i] = g[i] * (x[i] > 0);

	}
}

/**
 * gradient for activation use relu functions in cuda
 */
extern "C" void cacu_relu_grad_cuda(float_t *x, float_t *g, int length) {

	_k_CACU_RELU_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, g, length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_LEAKY_RELU_CUDA(float_t *x, float_t a, int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		x[i] = x[i] * a * (x[i] <= 0) + x[i] * (x[i] > 0);
	}

}

/**
 * for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_cuda(float_t *x, float_t a, int length) {

	_k_CACU_LEAKY_RELU_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, a, length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_LEAKY_RELU_GRAD_CUDA(float_t *x, float_t *g, float_t a,
		int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		g[i] = g[i] * a * (x[i] <= 0) + g[i] * (x[i] > 0);
	}

}

/**
 * gradient for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_leaky_relu_grad_cuda(float_t *x, float_t *g, float_t a,
		int length) {

	_k_CACU_LEAKY_RELU_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, g, a, length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_PRELU_CUDA(float_t *x, const float_t *slopes,
		const int num, const int channel, const int c_length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int length = num * channel * c_length;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		x[i] = x[i] * slopes[i / c_length % channel] * (x[i] <= 0)
				+ x[i] * (x[i] > 0);
	}

}

/**
 * for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_prelu_cuda(float_t *x, const float_t *slopes,
		const int num, const int channel, const int c_length) {

	_k_CACU_PRELU_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, slopes, num, channel,
			c_length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_PRELU_GRAD_CUDA(float_t *x, float_t *g,
		const float_t *slopes, float_t * g_slopes, const int num,
		const int channel, const int c_length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int length = num * channel * c_length;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		g[i] = g[i] * slopes[i / c_length % channel] * (x[i] <= 0)
				+ g[i] * (x[i] > 0);
		g_slopes[i / c_length % channel] += g[i] * x[i] * (x[i] <= 0);
	}

}

/**
 * gradient for activation use leaky_relu functions in cuda
 */
extern "C" void cacu_prelu_grad_cuda(float_t *x, float_t *g,
		const float_t *slopes, float_t * g_slopes, const int num,
		const int channel, const int c_length) {

	_k_CACU_PRELU_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, g, slopes, g_slopes,
			num, channel, c_length);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_SOFTMAX_CUDA(float_t *x, const int num,
		const int channel, const int width, const int height, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int length = width * height * num;
	int c_length = width * height;
	int p_length = channel * c_length;
	int index;

	__shared__ float_t sum[THREADNUM], max_data[THREADNUM];

	float_t *xp, *yp;

	for (int i = bid; i < length; i += BLOCKNUM) {

		index = i / c_length * p_length + i % c_length;
		xp = x + index;
		yp = y + index;

		if (tid == 0) {
			max_data[0] = xp[0];
			for (int c = 1; c < channel; c++)
				max_data[0] = max(max_data[0], xp[c * c_length]);
		}

		__syncthreads();

		//bank conflict
		max_data[tid] = max_data[0];

		sum[tid] = float_t(0);
		for (int c = tid; c < channel; c += THREADNUM) {
			yp[c * c_length] = exp(xp[c * c_length] - max_data[tid]);
			sum[tid] += yp[c * c_length];
		}

		__syncthreads();

		int acc_length = THREADNUM / 2;
		while (acc_length > 0) {
			if (tid < acc_length)
				sum[tid] += sum[tid + acc_length];
			acc_length /= 2;
			__syncthreads();
		}

		//bank conflict
		sum[tid] = sum[0];

		__syncthreads();

		for (int c = tid; c < channel; c += THREADNUM) {
			yp[c * c_length] /= sum[tid];
		}

	}

}

/**
 * for activation use softmax functions in cuda
 */
extern "C" void cacu_softmax_cuda(float_t *x, const int num, const int channel,
		const int width, const int height, float_t *y) {

	_k_CACU_SOFTMAX_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, num, channel, width,
			height, y);

	CUDA_CHECK(cudaThreadSynchronize());

}

__global__ void _k_CACU_TANH_CUDA(float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = std::tanh(x[i]);
	}

}

/**
 * gradient for activation use tanh functions in cuda
 */
extern "C" void cacu_tanh_cuda(float_t *x, int length, float_t *y) {
	_k_CACU_TANH_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_TANH_GRAD_CUDA(float_t *x, float_t *g, int length,
		float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = g[i] * (1 - x[i] * x[i]);
	}

}

/**
 * gradient for activation use half tanh functions in cuda
 */
extern "C" void cacu_tanh_grad_cuda(float_t *x, float_t *g, int length,
		float_t *y) {
	_k_CACU_TANH_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, g, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_HTANH_CUDA(float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = x[i] > 0 ? std::tanh(x[i]) : 0;
	}

}

/**
 * gradient for activation use half tanh functions in cuda
 */
extern "C" void cacu_htanh_cuda(float_t *x, int length, float_t *y) {
	_k_CACU_HTANH_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_HTANH_GRAD_CUDA(float_t *x, float_t *g, int length,
		float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i] > 0 ? g[i] * (float_t(1) - x[i] * x[i]) : 0;
	}

}

/**
 * gradient for activation use tanh functions in cuda
 */
extern "C" void cacu_htanh_grad_cuda(float_t *x, float_t *g, int length,
		float_t *y) {
	_k_CACU_HTANH_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, g, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SIGMOID_CUDA(float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = float_t(1) / (float_t(1) + exp(-x[i]));
	}

}

/**
 * gradient for activation use tanh functions in cuda
 */
extern "C" void cacu_sigmoid_cuda(float_t *x, int length, float_t *y) {
	_k_CACU_SIGMOID_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SIGMOID_GRAD_CUDA(float_t *x, float_t *g, int length,
		float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = g[i] * x[i] * (float_t(1) - x[i]);
	}

}

/**
 * gradient for activation use tanh functions in cuda
 */
extern "C" void cacu_sigmoid_grad_cuda(float_t *x, float_t *g, int length,
		float_t *y) {
	_k_CACU_SIGMOID_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, g, length, y);

	CUDA_CHECK(cudaThreadSynchronize());
}

}

#endif
#endif
