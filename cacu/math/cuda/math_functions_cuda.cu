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
#include "math_functions_cuda.h"

#if __USE_DEVICE__ == ON
#if __PARALLELTYPE__ == __CUDA__

#include "../../tensor/cuda/cuda_log.h"
#include <curand.h>

namespace cacu {

__global__ void _k_CACU_SAXPY_ATOMIC_CUDA(float_t *x, float_t a, float_t *y,
		const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		atomicAdd(y + i, a * x[i]);
	}

}

extern "C" void cacu_saxpy_atomic_cuda(float_t *x, float_t a, float_t *y,
		const int length) {
	_k_CACU_SAXPY_ATOMIC_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, a, y, length);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_ISAXB_CUDA(float_t *x, const int channel, const int width, const int height, const float_t a,
		int *index_, const float_t b, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
	float_t *xp,*yp;
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
extern "C" void cacu_isaxb_cuda(float_t *x, const int channel, const int width, const int height, const float_t a,
		int *index_, const float_t b, float_t *y) {

	_k_CACU_ISAXB_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, channel, width, height, a, index_, b, y);

	CUDA_CHECK(cuda_device_sync_status());

}

__global__ void _k_ARGMAX_CUDA(float_t *x, const int length,
		unsigned int *index_) {

	__shared__ float_t shared_data[THREADNUM];

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

extern "C" void cacu_argmax_cuda(float_t *x, const int length,
		unsigned int *index_) {
	_k_ARGMAX_CUDA<<<1, THREADNUM, 0>>>(x, length, index_);

	CUDA_CHECK(cuda_device_sync_status());
}

//__global__ void _k_CACU_TRANSPOSE_CUDA(float_t *mtx, const int m, const int n,
//		const int clength) {
//
//	int tid = threadIdx.x;
//	int bid = blockIdx.x;
//
//	__shared__ float_t shared_data[THREADNUM];
//
//	int next, cur, pre;
//
//	for (int c = bid; c < clength; c+=BLOCKNUM) {
//		for (int i = tid; i < m * n; i+=THREADNUM) {
//			next = (i % n) * m + i / n;
//			while (next > i)
//			next = (next % n) * m + next / n;
//			if (next == i) {
//				shared_data[tid] = mtx[i*clength + c];
//				cur = i;
//				pre = (cur % m) * n + cur / m;
//				while (pre != i) {
//					mtx[cur*clength + c] = mtx[pre*clength + c];
//					cur = pre;
//					pre = (cur % m) * n + cur / m;
//				}
//				mtx[cur*clength + c] = shared_data[tid];
//			}
//		}
//	}
//}

__global__ void _k_CACU_TRANSPOSE_CUDA(float_t *mtx, const int m, const int n,
		const int clength) {
		
	int tid = threadIdx.x;
	int bid = blockIdx.x;
		
	__shared__ float_t shared_data[THREADNUM];
	
	int next, cur, pre;

	for (int c = bid; c < clength; c+=BLOCKNUM) {
		for (int i = tid; i < m * n; i+=THREADNUM) {
			next = (i % n) * m + i / n;
			while (next > i)
			next = (next % n) * m + next / n;
			if (next == i) {
				shared_data[tid] = mtx[i*clength + c];
				cur = i;
				pre = (cur % m) * n + cur / m;
				while (pre != i) {
					mtx[cur*clength + c] = mtx[pre*clength + c];
					cur = pre;
					pre = (cur % m) * n + cur / m;
				}
				mtx[cur*clength + c] = shared_data[tid];
			}
		}
	}
}


extern "C" void cacu_transpose_cuda(float_t *mtx, const int m, const int n, const int clength) {
	
	_k_CACU_TRANSPOSE_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(mtx, m, n, clength);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_CLIP_VEC_CUDA(float_t *data, const float_t l_t, const float_t r_t,
		const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		data[i] = data[i] >= l_t ? data[i] : l_t;
		data[i] = data[i] <= r_t ? data[i] : r_t;
	}
}

extern "C" void cacu_clip_vec_cuda(float_t *data, const float_t l_t, const float_t r_t,
		const int length) {
	_k_CACU_CLIP_VEC_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(data, l_t, r_t, length);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_ABS_CUDA(float_t *x, const int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = abs(x[i]);
	}
}

extern "C" void cacu_abs_cuda(float_t *x, const int length, float_t *y)
{
	_k_CACU_ABS_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_ABS_GRAD_CUDA(float_t *x, float_t *diff, const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		diff[i] = (x[i] >= 0) ? 1 : -1;
	}
}

extern "C" void cacu_abs_grad_cuda(float_t *x, float_t *diff, const int length)
{
	_k_CACU_ABS_GRAD_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, diff, length);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_LN_CUDA(float_t *x, const int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = logf(x[i]);
	}
}
extern "C" void cacu_ln_cuda(float_t *x, const int length, float_t *y){

	_k_CACU_LN_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_POW_CUDA(float_t *x, const int length, const float_t pow, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = powf(x[i], pow);
	}
}

extern "C" void cacu_pow_cuda(float_t *x, const int length, const float_t pow, float_t *y){

	_k_CACU_POW_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, pow, y);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_POWT_CUDA(float_t *x, const int length, const float_t *pow, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = powf(x[i], pow[i]);
	}
}

extern "C" void cacu_powt_cuda(float_t *x, const int length, const float_t *pow, float_t *y){

	_k_CACU_POWT_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, pow, y);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_FLIP_CUDA(int *x, const int length) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		x[i] = (x[i] == 0);
	}
}
extern "C" void cacu_flip_cuda(int *x, const int length){

	_k_CACU_FLIP_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length);

	CUDA_CHECK(cuda_device_sync_status());
}


__global__ void _k_CACU_ACOS_CUDA(float_t *x, const int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = acosf(x[i]);
	}
}

extern "C" void cacu_acos_cuda(float_t *x, const int length, float_t *y){

	_k_CACU_ACOS_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cuda_device_sync_status());
}

__global__ void _k_CACU_COS_CUDA(float_t *x, const int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = cosf(x[i]);
	}
}

extern "C" void cacu_cos_cuda(float_t *x, const int length, float_t *y){

	_k_CACU_COS_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cuda_device_sync_status());
}


__global__ void _k_CACU_FLOOR_CUDA(float_t *x, const int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {

		y[i] = floorf(x[i]);
	}
}

extern "C" void cacu_floor_cuda(float_t *x, const int length, float_t *y){

	_k_CACU_FLOOR_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);

	CUDA_CHECK(cuda_device_sync_status());
}

extern "C" void cacu_rotate_cuda(float_t *mtx, const int m, const int n, const int clength, rotate_code rotate){

	CUDA_CHECK(cuda_device_sync_status());
}
//
//__device__ float _GENERATE_RAND(curandState* global_state_, int index){
//
//	curandState local_state = global_state[index];
//	float RANDOM = curand_uniform(&local_state);
//	global_state[index] = local_state;
//	return RANDOM;
//
//}

__global__ void _k_RAND_VECTOR_CUDA(float_t *vector_, const int length, const float_t ratio_) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;
//	float_t pRandomValue;


	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
//		pRandomValue = ((float_t) rand() / (float_t) RAND_MAX);
//		pRandomValue = pRandomValue * (1.0 - 0.0) + 0.0;
//		vector_[i] = pRandomValue >= ratio_;
	}
}

extern "C" void rand_vector_cuda(float_t *vector_, const int length, const float_t ratio_){

	_k_RAND_VECTOR_CUDA<<<BLOCKNUM, THREADNUM, 0>>>(vector_, length, ratio_);
	CUDA_CHECK(cuda_device_sync_status());
}


}

#endif
#endif
