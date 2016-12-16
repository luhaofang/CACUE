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
#include "../math_utils.h"


__global__ void _k_CACU_SUMBYSIZE_BYWIDTH_GPU(float_t *x, int heigth, int width, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	float_t *start;

	for (int i = threadid; i < heigth; i += BLOCKNUM * THREADNUM) {
		start = x + i * width;
		for(int j = 0 ;  j < width; ++j)
			y[i] += start[j];
	}
}

__global__ void _k_CACU_SUMBYSIZE_BYHEIGHT_GPU(float_t *x, int height, int width, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	float_t *start;

	for (int i = threadid; i < width; i += BLOCKNUM * THREADNUM) {
		start = x + i;
		for(int j = 0 ;  j < height; ++j)
			y[i] += start[j*width];
	}
}

extern "C" void cacu_sumbysize_gpu(SUM SUMTYPE ,float_t *x, int length, float_t *y, int width){

	int height = length / width;

	if (BYWIDTH == SUMTYPE)
		_k_CACU_SUMBYSIZE_BYWIDTH_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, height,width, y);
	else if(BYHEIGHT == SUMTYPE)
		_k_CACU_SUMBYSIZE_BYHEIGHT_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, height,width, y);
	CUDA_CHECK(cudaThreadSynchronize());
}


extern "C" void cacu_cxsize_gpu(float_t *x, int length, float_t *a, int size,float_t *y);

extern "C" void cacu_sxsize_gpu(float_t *x, int length, float_t a, float_t *y);

extern "C" void cacu_cdxsize_gpu(float_t *x, int length, float_t *a, int size, float_t *y);

extern "C" void cacu_sdxsize_gpu(float_t *x, int length, float_t a, float_t *y);

__global__ void _k_CACU_SSXPY_GPU(float_t *x, float_t a, int size, float_t *y, float_t b, int length, float_t *z) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int block_size = length / size;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		z[i] = b*y[i] + a*x[i / block_size];
	}
}


extern "C" void cacu_ssxpy_gpu(float_t *x, float_t a, int size, float_t *y, float_t b, int length, float_t *z)
{
	_k_CACU_SSXPY_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, a, size, y, b, length, z);
	CUDA_CHECK(cudaThreadSynchronize());
}

extern "C" void cacu_sqr_gpu(float_t *x, int length, float_t *y);

extern "C" void cacu_root_gpu(float_t *x, int length, float_t *y);

extern "C" void cacu_stdbychannel_gpu(float_t *x, int length, float_t *y, float_t epsilon);

extern "C" void cacu_ssx_gpu(float_t *x, int length, float_t *y, float_t epsilon);
