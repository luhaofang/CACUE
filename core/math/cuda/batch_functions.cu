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

	float_t *start;

	extern __shared__ float_t shared_data[];

	for (int i = bid; i < heigth; i += BLOCKNUM) {
		start = x + i * width;
		for(int j = tid ;  j < width; j += THREADNUM)
			shared_data[tid] += start[j];
		__syncthreads();

		if(tid == 0){
			for(int j = 0; j < THREADNUM ; ++j)
				shared_data[0] += shared_data[j];
			y[i] = shared_data[0];
		}
	}
}

__global__ void _k_CACU_SUMBYSIZE_BYHEIGHT_GPU(float_t *x, int height, int width, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	float_t *start;

	extern __shared__ float_t shared_data[];

	for (int i = bid; i < width; i += BLOCKNUM) {

		start = x + i;
		for(int j = tid ;j < height; j += THREADNUM)
			shared_data[tid] += start[j*width];
		__syncthreads();

		if(tid == 0){
			for(int j = 0; j < THREADNUM ; ++j)
				shared_data[0] += shared_data[j];
			y[i] = shared_data[0];
		}
	}
}

/**
 * @cacu_sumbysize_gpu
 * sum by size:
 * accumulate the value by width or height , width is the matrix array's width dim which stored in row -major format.
 * sum by width y is (length/ width) height dim, sum by height y is width dim.
 */
extern "C" void cacu_sumbysize_gpu(SUM SUMTYPE ,float_t *x, int length, float_t *y, int width){

	int height = length / width;

	if (BYWIDTH == SUMTYPE)
		_k_CACU_SUMBYSIZE_BYWIDTH_GPU<<<height, width, width>>>(x, height,width, y);
	else if(BYHEIGHT == SUMTYPE)
		_k_CACU_SUMBYSIZE_BYHEIGHT_GPU<<<width, height, height>>>(x, height,width, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_CXSIZE_GPU(float_t *x, int length, float_t *a, int size,float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int block_size = length / size;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i]*a[i / block_size];
	}
}

/**
 * @cacu_cxsize_gpu
 * math y[i] = a[j]*x[i] :
 * x: length dim array list
 * a: size dim array list
 * a[j] is the corresponding scalar, j = i / (length / size).
 */
extern "C" void cacu_cxsize_gpu(float_t *x, int length, float_t *a, int size,float_t *y)
{
	_k_CACU_CXSIZE_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, length, a, size, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_CDXSIZE_GPU(float_t *x, int length, float_t *a, int size ,float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	int block_size = length / size;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i]/a[i / block_size];
	}
}

/**
 * @cacu_cdxsize_gpu
 * math y[i] = x[i] / a[j] :
 * x: length dim array list
 * a: size dim array list
 * a[j] is the corresponding denominator, j = i / (length / size).
 */
extern "C" void cacu_cdxsize_gpu(float_t *x, int length, float_t *a, int size, float_t *y)
{
	_k_CACU_CDXSIZE_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, length, a, size, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

/**
 * @cacu_sdxsize_gpu
 * math y[i] = x[i] / a :
 * x: length dim array list
 * a: the corresponding denominator.
 */
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

/**
 * @cacu_ssxpy_gpu
 * math z[i] = a * x[j] + b * y[i] :
 * y: length dim array list
 * x: size dim array list
 * x[j] is the corresponding scalar, j = i / (length / size).
 * a & b are corresponding scalars for x, y
 */
extern "C" void cacu_ssxpy_gpu(float_t *x, float_t a, int size, float_t *y, float_t b, int length, float_t *z)
{
	_k_CACU_SSXPY_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, a, size, y, b, length, z);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_SQR_GPU(float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = x[i] * x[i];
	}
}

/**
 * @cacu_sqr_gpu
 * math y[i] = x[i]^2 :
 */
extern "C" void cacu_sqr_gpu(float_t *x, int length, float_t *y)
{
	_k_CACU_SQR_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_ROOT_GPU(float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] = sqrt(x[i]);
	}
}

/**
 * @cacu_root_gpu
 * math y[i] = sqrt(x[i]) :
 */
extern "C" void cacu_root_gpu(float_t *x, int length, float_t *y)
{
	_k_CACU_ROOT_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}

__global__ void _k_CACU_STDBYCHANNEL_GPU(float_t *varience, int length, float_t *std, float_t epsilon) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		std[i] = sqrt(varience[i] + epsilon);
	}
}

/**
 * @cacu_stdbychannel_gpu
 * math std[i] = sqrt(varience[i] + epsilon) :
 */
extern "C" void cacu_stdbychannel_gpu(float_t *varience, int length, float_t *std, float_t epsilon)
{
	_k_CACU_STDBYCHANNEL_GPU<<<BLOCKNUM, THREADNUM, 0>>>(varience, length, std, epsilon);
	CUDA_CHECK(cudaThreadSynchronize());
}


__global__ void _k_CACU_SSX_GPU(float_t *x, int length, float_t *y) {

	int tid = threadIdx.x;
	int bid = blockIdx.x;

	int threadid = bid * THREADNUM + tid;

	for (int i = threadid; i < length; i += BLOCKNUM * THREADNUM) {
		y[i] *= x[i];
	}
}

/**
 * @cacu_ssx_gpu
 * math y[i] *= x[i] :
 * scale by element wise.
 */
extern "C" void cacu_ssx_gpu(float_t *x, int length, float_t *y)
{
	_k_CACU_SSX_GPU<<<BLOCKNUM, THREADNUM, 0>>>(x, length, y);
	CUDA_CHECK(cudaThreadSynchronize());
}
