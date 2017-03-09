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

#pragma once


#include "cuda_log.h"
#include "cuda_utils.h"

#if __PARALLELTYPE__ == __GPU__


inline void cacu_saxpy_gpu(float_t *x, float_t a, float_t *y, int length) {
	status = cublasSaxpy_v2(handle, length, &a, x, 1, y, 1);
	CUBLAS_CHECK(status);
}

inline void cacu_saxpby_gpu(float_t *x, float_t a, float_t *y, float_t b, int length)
{
	status = cublasSscal_v2(handle, length, &a, y, 1);
	CUBLAS_CHECK(status);
	status = cublasSaxpy_v2(handle, length, &a, x, 1, y, 1);
	CUBLAS_CHECK(status);
}

inline void cacu_scalex_gpu(float_t *x, float_t a, int length)
{
	status = cublasSscal_v2(handle, length, &a, x, 1);
	CUBLAS_CHECK(status);
}

inline void cacu_sgemv_gpu(cublasOperation_t trans, float_t *x, int x_height, float_t *y, int x_width, float_t *z)
{
	float_t alpha = 1;
	float_t beta = 0;
	status = cublasSgemv_v2(handle, trans, x_height, x_width, &alpha, x, x_height, y, 1, &beta, z, 1);
	CUBLAS_CHECK(status);
}

inline void cacu_sgemm_gpu(cublasOperation_t transx, cublasOperation_t transy, float_t *x, int x_height, int x_width, float_t *y, int y_width, float_t alpha, float_t *z, float_t beta)
{
	int m = x_height,n = y_width,k = x_width;
	int lda = (transx == CUBLAS_OP_N) ? m : k;
	int ldb = (transy == CUBLAS_OP_N) ? k : n;
	float_t _alpha = alpha;
	float_t _beta = beta;
	status = cublasSgemm_v2(handle, transx, transy, m, n, k, &alpha, x, lda, y, ldb, &beta, z, m);
	CUBLAS_CHECK(status);
}

inline void cacu_copy_gpu(float_t *x, int x_length,float_t *y)
{
	status = cublasScopy_v2(handle, x_length, x, 1, y, 1);
	CUBLAS_CHECK(status);
}

extern "C" void cacu_isaxb_gpu(float_t *x, int length, float_t a ,unsigned int *index_, float_t b, float_t *y);


#endif
