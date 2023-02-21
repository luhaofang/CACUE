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
#include "norm_functions_cuda.h"

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

#include "../../definition.h"
#include "../../tensor/cuda/cuda_log.h"

namespace cacu{

__global__ void _k_CACU_NORMALIZATION_L1_CUDA(float_t *x, const int length, const float_t epsilon) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	shared_data[tid] = 0;
	for (int j = tid; j < length; j += THREADNUM) {
		shared_data[tid] += abs(x[j]);
	}
	__syncthreads();

	int acc_length = THREADNUM / 2;
	while (acc_length > 0) {
		if (tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	shared_data[tid] = shared_data[0] + epsilon;

	__syncthreads();

	for (int i = tid; i < length; i += THREADNUM) {
		x[i] /= shared_data[tid];
	}
}


extern "C" void cacu_normalization_l1_cuda(float_t *x, const int length, const float_t epsilon) {

	_k_CACU_NORMALIZATION_L1_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, length, epsilon);

	CUDA_CHECK(cuda_device_sync_status());

}

__global__ void _k_CACU_NORMALIZATION_L1_GRAD_CUDA(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	shared_data[tid] = 0;
	for (int j = tid; j < length; j += THREADNUM) {
		shared_data[tid] += abs(x[j]);
	}
	__syncthreads();

	int acc_length = THREADNUM / 2;
	while (acc_length > 0) {
		if (tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	shared_data[tid] = shared_data[0] + epsilon;

	__syncthreads();

	for (int i = tid; i < length; i += THREADNUM) {
		xg[i] = (shared_data[tid] - x[i]) / powf(shared_data[tid], 2.0) * yg[i];
	}
}


extern "C" void cacu_normalization_l1_grad_cuda(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {

	_k_CACU_NORMALIZATION_L1_GRAD_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, length, epsilon, yg, xg);

	CUDA_CHECK(cuda_device_sync_status());

}


__global__ void _k_CACU_NORMALIZATION_L2_CUDA(float_t *x, const int length, const float_t epsilon) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	shared_data[tid] = 0;
	for (int j = tid; j < length; j += THREADNUM) {
		shared_data[tid] += x[j]*x[j];
	}
	__syncthreads();

	int acc_length = THREADNUM / 2;
	while (acc_length > 0) {
		if (tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	shared_data[tid] = sqrtf(shared_data[0]) + epsilon;

	__syncthreads();

	for (int i = tid; i < length; i += THREADNUM) {
		x[i] /= shared_data[tid];
	}
}

extern "C" void cacu_normalization_l2_cuda(float_t *x, const int length, const float_t epsilon) {

	_k_CACU_NORMALIZATION_L2_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, length, epsilon);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_NORMALIZATION_L2_GRAD_CUDA(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	shared_data[tid] = 0;
	for (int j = tid; j < length; j += THREADNUM) {
		shared_data[tid] += abs(x[j]);
	}
	__syncthreads();

	int acc_length = THREADNUM / 2;
	while (acc_length > 0) {
		if (tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	shared_data[tid] = shared_data[0] + epsilon;

	__syncthreads();

	for (int i = tid; i < length; i += THREADNUM) {
		xg[i] = (powf(shared_data[tid], 2.0) - powf(x[i], 2.0)) / powf(shared_data[tid], 3.0) * yg[i];
	}
}


extern "C" void cacu_normalization_l2_grad_cuda(float_t *x, const int length, const float_t epsilon, float_t *yg, float_t *xg) {

	_k_CACU_NORMALIZATION_L2_GRAD_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, length, epsilon, yg, xg);

	CUDA_CHECK(cuda_device_sync_status());

}

__global__ void _k_CACU_NORM_L1_CUDA(float_t *x, const int length, const float_t epsilon, float_t *norm) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	shared_data[tid] = 0;
	for (int j = tid; j < length; j += THREADNUM) {
		shared_data[tid] += abs(x[j]);
	}
	__syncthreads();

	int acc_length = THREADNUM / 2;
	while (acc_length > 0) {
		if (tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	if(tid == 0)
		norm[0] = shared_data[0] + epsilon;
}

extern "C" void cacu_norm_l1_cuda(float_t *x, const int length, const float_t epsilon, float_t *norm){
	_k_CACU_NORM_L1_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, length, epsilon, norm);
	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_NORM_L2_CUDA(float_t *x, const int length, const float_t epsilon, float_t *norm) {

	int tid = threadIdx.x;

	extern __shared__ float_t shared_data[];

	shared_data[tid] = 0;
	for (int j = tid; j < length; j += THREADNUM) {
		shared_data[tid] += x[j]*x[j];
	}
	__syncthreads();

	int acc_length = THREADNUM / 2;
	while (acc_length > 0) {
		if (tid < acc_length)
			shared_data[tid] += shared_data[tid + acc_length];
		acc_length /= 2;
		__syncthreads();
	}

	if(tid == 0)
		norm[0] = sqrtf(shared_data[0]) + epsilon;

}

extern "C" void cacu_norm_l2_cuda(float_t *x, const int length, const float_t epsilon, float_t *norm){
	_k_CACU_NORM_L2_CUDA<<<1, THREADNUM, THREADNUM * sizeof(float_t)>>>(x, length, epsilon, norm);
	CUDA_CHECK(cuda_device_sync_status());
}

}

#endif
#endif

