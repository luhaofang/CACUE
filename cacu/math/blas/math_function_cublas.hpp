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

#ifdef __PARALLELTYPE__
#if __PARALLELTYPE__ == __CUDA__

#include "cublas_utils.h"

namespace cacu{

inline void cacu_saxpy_cublas(float *x, const float a, float *y, const int length) {

	CUBLAS_CHECK(cublasSaxpy_v2(handle, length, &a, x, 1, y, 1));
}

inline void cacu_saxpby_cublas(float *x, const float a, float *y, const float b, const int length)
{
	CUBLAS_CHECK(cublasSscal_v2(handle, length, &b, y, 1));
	CUBLAS_CHECK(cublasSaxpy_v2(handle, length, &a, x, 1, y, 1));
}

inline void cacu_scalex_cublas(float *x, const float a, const int length)
{
	CUBLAS_CHECK(cublasSscal_v2(handle, length, &a, x, 1));
}

inline void cacu_sgemv_cublas(cublasOperation_t trans,float *x, const int x_height,float *y, const int x_width,const float alpha, float *z ,const float beta)
{
	int m = x_height,n = x_width;
	CUBLAS_CHECK(cublasSgemv_v2(handle, trans, m, n, &alpha, x, m, y, 1, &beta, z, 1));
}

inline void cacu_sgemm_cublas(cublasOperation_t transx, cublasOperation_t transy, float *x, const int x_height, const int x_width,float *y, const int y_width, const float alpha, float *z,const float beta)
{
	int m = x_height,n = y_width,k = x_width;
	int lda = (transx == CUBLAS_OP_N) ? m : k;
	int ldb = (transy == CUBLAS_OP_N) ? k : n;
	CUBLAS_CHECK(cublasSgemm_v2(handle, transx, transy, m, n, k, &alpha, x, lda, y, ldb, &beta, z, m));
}

inline void cacu_copy_cublas(const float *x, int length, float *y){
	CUBLAS_CHECK(cublasScopy_v2(handle, length, x, 1, y, 1));
}


}

#endif
#endif
